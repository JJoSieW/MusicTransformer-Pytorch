import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from third_party.midi_processor.processor import Event  # 确保你能从 token id 解出 Event


# Borrowed from https://github.com/jason9693/MusicTransformer-pytorch/blob/5f183374833ff6b7e17f3a24e3594dedd93a5fe5/custom/criterion.py#L28
class SmoothCrossEntropyLoss(_Loss):
    """
    https://arxiv.org/abs/1512.00567
    """
    __constants__ = ['label_smoothing', 'vocab_size', 'ignore_index', 'reduction']

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, reduction='mean', is_logits=True):
        assert 0.0 <= label_smoothing <= 1.0
        super().__init__(reduction=reduction)

        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.input_is_logits = is_logits
        
        # 处理ignore_index为列表的情况
        if isinstance(ignore_index, (list, tuple)):
            self.ignore_indices = ignore_index
        else:
            self.ignore_indices = [ignore_index]

    def forward(self, input, target):
        """
        Args:
            input: [B * T, V]  --- B: batch size, T: sequence length, V: vocabulary size
            target: [B * T]
        Returns:
            cross entropy: [1]
        """        
        # 创建掩码：标记哪些位置是需要忽略的token
        # mask = (target == self.ignore_index).unsqueeze(-1)
        mask = torch.zeros_like(target, dtype=torch.bool)
        for ignore_idx in self.ignore_indices:
            mask |= (target == ignore_idx)
        mask = mask.unsqueeze(-1)
    
        q = F.one_hot(target.long(), self.vocab_size).type(torch.float32)
        u = 1.0 / self.vocab_size
        q_prime = (1.0 - self.label_smoothing) * q + self.label_smoothing * u
        q_prime = q_prime.masked_fill(mask, 0)

        ce = self.cross_entropy_with_logits(q_prime, input)
        if self.reduction == 'mean':
            lengths = torch.sum(~mask.squeeze(-1))  # 非ignore_index的token的数量
            return ce.sum() / lengths
        elif self.reduction == 'sum':
            return ce.sum()
        else:
            raise NotImplementedError

    def cross_entropy_with_logits(self, p, q):
        return -torch.sum(p * (q - q.logsumexp(dim=-1, keepdim=True)), dim=-1)



# The following classes are new additions by https://github.com/JJoSieW.
class ContourLoss(nn.Module):
    def __init__(self, margin=0.87):
        super().__init__()
        self.margin = margin
        
    def forward(self, token_batch):
        """
        Args:
            token_batch: Tensor of shape (B, T)
        Returns:
            scalar contour-aware loss
        """
        total_loss = 0.0
        num_pairs = 0
        B = token_batch.size(0)

        for b in range(B):
            tokens = token_batch[b]
            event_sequence = [Event.from_int(idx.item()) for idx in tokens]
            contour_blocks = self.extract_contour_note_pairs(event_sequence)

            for pair in contour_blocks:
                note_vecs = pair.get("note_vectors", [])
                if len(note_vecs) <= 1:
                    continue

                note_vecs = torch.tensor(note_vecs, dtype=torch.float32, device=token_batch.device)
                contour_vec = torch.tensor(
                    [pair["contour_duration"], pair["contour_interval"]],
                    dtype=torch.float32,
                    device=token_batch.device
                )
                loss = self.cosine_margin_loss(note_vecs, contour_vec)
                total_loss += loss
                num_pairs += 1

        if num_pairs == 0:
            return torch.tensor(0.0, device=token_batch.device)

        return total_loss / num_pairs 
    
    
    def extract_contour_note_pairs(self, event_sequence, max_len=None):
        """
        Extract contour-note pairs from a sequence of Events.
        Returns list of dicts with contour_interval, contour_duration, note_on_pitches, note_vectors, start_idx, end_idx
        """
        pairs = []
        i = 0
        while i < len(event_sequence):
            if max_len is not None and i >= max_len:
                break

            event = event_sequence[i]
            if event.type == "contour_interval":
                if i + 1 >= len(event_sequence):
                    print(f"[extract warning] contour_interval at {i} not followed by contour_duration")
                    i += 1
                    continue
                elif event_sequence[i+1].type != "contour_duration":
                    print(f"[extract warning] contour_interval at {i} not followed by contour_duration")
                    i += 1
                    continue

                contour_interval = event.value
                contour_duration = event_sequence[i+1].value
                time_passed = 0
                note_pitches = []
                note_vectors = []
                first_note_time = None
                first_note_pitch = None

                # 收集这段 contour_duration 内所有 note_on
                notes_in_segment = []
                j = i + 2
                time_passed = 0
                while j < len(event_sequence) and time_passed < contour_duration:
                    e = event_sequence[j]
                    if e.type == "time_shift":
                        time_passed += e.value
                    elif e.type == "note_on":
                        notes_in_segment.append((time_passed, e.value))  # (time, pitch)
                    j += 1

                if not notes_in_segment:
                    i = j
                    continue

                # 分成 5 个时间段（均匀划分）
                slice_num = 5
                interval_len = contour_duration / slice_num
                selected_vectors = []
                first_time, first_pitch = notes_in_segment[0]

                for k in range(slice_num):
                    t_start = k * interval_len
                    t_end = (k + 1) * interval_len
                    candidates = [note for note in notes_in_segment if t_start <= note[0] < t_end]
                    if not candidates:
                        continue
                    max_note = max(candidates, key=lambda x: x[1])  # 取最高 pitch
                    dt = max_note[0] - first_time
                    dp = max_note[1] - first_pitch
                    selected_vectors.append((dt, dp))

                if not selected_vectors:
                    i = j
                    continue

                pairs.append({
                    "contour_interval": contour_interval,
                    "contour_duration": contour_duration,
                    "note_vectors": selected_vectors,
                    "start_idx": i,
                    "end_idx": j
                })

                i = j
            else:
                i += 1

        return pairs

    def cosine_margin_loss(self, note_vectors, contour_vector):
        if note_vectors.size(0) <= 1:
            return torch.tensor(0.0, device=note_vectors.device)

        contour_vec = F.normalize(contour_vector, dim=0)  # shape (2,)
        note_vecs = F.normalize(note_vectors, dim=1)      # shape (N, 2)
        cos_sim = torch.matmul(note_vecs, contour_vec)    # shape (N,)
        loss = torch.clamp(self.margin - cos_sim, min=0.0)
        return loss.mean()




class CombinedLoss(nn.Module):
    def __init__(self, ce_loss_fn, contour_loss_fn, lambda_contour_scheduler):
        """
        Combines cross entropy loss and contour-aware loss.

        Args:
            ce_loss_fn: nn.Module, e.g., SmoothCrossEntropyLoss
            contour_loss_fn: nn.Module, e.g., ContourLoss
            lambda_contour: float, scaling factor for contour loss
        """
        super().__init__()
        self.ce_loss_fn = ce_loss_fn
        self.contour_loss_fn = contour_loss_fn
        self.lambda_scheduler = lambda_contour_scheduler
        self.cur_epoch = 0  # will be updated externally
        
    def set_epoch(self, epoch):
        self.cur_epoch = epoch

    def forward(self, logits, targets):
        """
        Args:
            logits: [B * T, V] — flattened logits
            targets: [B, T] — original token IDs

        Returns:
            Scalar loss
        """
        logits = logits.reshape(logits.shape[0] * logits.shape[1], -1)  # [B*T, V] 
        tgt_flat = targets.flatten()                                    # [B*T]      

        ce_loss = self.ce_loss_fn(logits, tgt_flat)
        contour_loss = self.contour_loss_fn(targets)

        lambda_contour = self.lambda_scheduler(self.cur_epoch)
        return ce_loss + lambda_contour * contour_loss