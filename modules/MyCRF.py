# -*- coding: utf-8 -*-
# @Time    : 2020/12/25
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : MyCRF.py
import torch
import torch.nn as nn

class MyCRF(nn.Module):
    def __init__(self,num_tags,constraints=None,include_start_end_transitions=False):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.include_start_end_transitions=include_start_end_transitions
        self.reset_parameters()

    def forward(self,inputs, tags, mask):
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool)
        else:
            mask = mask.to(torch.bool)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        log_denominator = self._input_likelihood(inputs, mask)
        res=torch.sum(log_numerator-log_denominator)
        return res

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """计算分母"""
        batch_size, sequence_length, num_tags = logits.size()

        # 转换维度方便进行批处理和矩阵运算
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        #
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]


        # 计算每一步的所有可能的分数
        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # emit_scores[0]：[1,num_tags]
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # transition_scores[0]：[num_tags,num_tags]
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            # broadcast_alpha[0]：[num_tags,1]
            inner = broadcast_alpha + emit_scores + transition_scores
            # inner [num_tags, num_tags]

            alpha = self.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (
                ~mask[i]
            ).view(batch_size, 1)

        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        return self.logsumexp(stops)

    def _joint_likelihood(
            self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """计算分子"""
        batch_size, sequence_length, _ = logits.data.shape
        # 转置前两维是为了能够进行批处理
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # 如果输入不包含开始和结束约束的话，初始分数即为 0
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # 计算到最后一个字符的总得分，注意当前计算的分数是下一个字符的总分数，所以只循环sequence_length-1次
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            # 如果当前字符被 mask 了，当前的分数就等于上一个未被 mask 的分数
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # [start_tag, 0, 1, 0, 1,...,0, end_tag]
        # 上边循环只计算了end_tag之前一步的总分数，我们需要到end_tag的分数，先找end_tag的下标，然后计算最后得分
        # 找下标
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # 获取最后一步的转移分数
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # 获取最后一步的发射分数
        last_inputs = logits[-1]  # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()  # (batch_size,)
        # 计算总得分
        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def logsumexp(self,tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
        """
        A numerically stable computation of logsumexp. This is mathematically equivalent to
        `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
        probabilities.

        # Parameters

        tensor : `torch.FloatTensor`, required.
            A tensor of arbitrary size.
        dim : `int`, optional (default = `-1`)
            The dimension of the tensor to apply the logsumexp to.
        keepdim: `bool`, optional (default = `False`)
            Whether to retain a dimension of size one at the dimension we reduce over.
        """
        max_score, _ = tensor.max(dim, keepdim=keepdim)
        if keepdim:
            stable_vec = tensor - max_score
        else:
            stable_vec = tensor - max_score.unsqueeze(dim)
        tmp=(stable_vec.exp().sum(dim, keepdim=keepdim)).log()
        res=max_score + tmp
        return res

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)

    def viterbi_tags(self,logits,mask):
        # logits:[batch,max_seq_len,num_tags]
        batch, max_seq_length, num_tags = logits.size()  # 这里的num_tags和self.num_tags是一样的

        # 转移矩阵：transition_matrix: [num_tags+2,num_tags+2] ,加2是因为加了一个开始标签和一个结束标签，
        # transition_matrix[i,j]表示从标签 i 变到 标签 j 的概率是多少
        transition_matrix=torch.Tensor(num_tags+2,num_tags+2).fill_(-10000.0)
        start_tag=0 # 开始标签在trainsition中的下标为 0
        end_tag=num_tags+1 # 结束标签为 最后一个下标
        transition_matrix[start_tag+1:end_tag,start_tag+1:end_tag]=self.transitions # 将学习到的转移概率赋值
        transition_matrix[start_tag,start_tag+1:end_tag]=torch.zeros(1,end_tag-start_tag-1)
        transition_matrix[start_tag+1:end_tag,end_tag]=torch.zeros(end_tag-start_tag-1)

        # 发射矩阵：tag_sequence:[max_seq_length+2, num_tags+2],max_seq_length+2表示加了一个开始字符和一个结束字符。
        # tag_sequence[i,j]表示出现字符 i， 其标签为 j 的概率是多少

        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)
        best_paths=[]
        # 每一个循环求解batch中的一个解
        for prediction, prediction_mask in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]

            tag_sequence.fill_(-10000.0)
            tag_sequence[0,start_tag]=0.0
            tag_sequence[1:sequence_length+1,start_tag+1:end_tag]=masked_prediction
            tag_sequence[sequence_length+1,end_tag]=0.0
            # ([[0, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2, tensor(3)]], tensor([53.2492]))
            viterbi_paths, viterbi_scores = self.viterbi_decode(
                tag_sequence=tag_sequence[: (sequence_length + 2)],
                transition_matrix=transition_matrix,
            )
            paths=[]
            for path in viterbi_paths:
                paths.append([i-1 for i in path[1:-1]])
            best_paths.append((paths[0], viterbi_scores.item()))
        return best_paths
    def viterbi_decode(self,tag_sequence,transition_matrix,top_k=1):
        sequence_length, num_tags = list(tag_sequence.size())
        path_scores = []
        path_indices = []
        path_scores.append(tag_sequence[0, :].unsqueeze(0))
        for timestep in range(1, sequence_length):
            sum_score_t=path_scores[timestep-1].unsqueeze(2)+transition_matrix
            sum_score_t=sum_score_t.view(-1,num_tags)
            max_k=min(sum_score_t.size()[0], top_k)
            scores, paths = torch.topk(sum_score_t, k=max_k, dim=0)
            path_scores.append(tag_sequence[timestep, :] + scores)
            path_indices.append(paths.squeeze())
        path_scores_v = path_scores[-1].view(-1)
        max_k = min(path_scores_v.size()[0], top_k)
        viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
        viterbi_paths = []
        for i in range(max_k):
            viterbi_path = [best_paths[i]]
            for backward_timestep in reversed(path_indices):
                viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
            viterbi_path.reverse()
            viterbi_path = [j % num_tags for j in viterbi_path]
            viterbi_paths.append(viterbi_path)
        return viterbi_paths, viterbi_scores