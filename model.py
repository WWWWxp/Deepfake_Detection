import os, torch
from torch import nn
from safetensors.torch import load_file
# 用 fairseq 版 Wav2Vec2；如需 transformers 版自行替换
#from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from module.AASIST import AASIST
from transformers import AutoConfig, AutoModelForPreTraining


class SSLModel(nn.Module):
    def __init__(self, layers: int = 5):
        super().__init__()
        # 1) 先载入原始配置并覆写层数
        config = AutoConfig.from_pretrained(
            "ckpts/xlsr-300m",
            num_hidden_layers=layers           # ★ 提前写进去
        )

        # 2) 再用这个修改后的 config 创建模型，
        #    并允许尺寸不匹配的权重被自动忽略
        self.model = AutoModelForPreTraining.from_pretrained(
            "ckpts/xlsr-300m",
            config=config
        )
        self.model.config.output_hidden_states = True

    def forward(self, x):
        return self.model(x).hidden_states[5]  # (B, T', D)

    @property
    def out_dim(self):
        return self.model.config.hidden_size


# ---------------- SSL + AASIST ----------------
class Wav2Vec2_AASIST(nn.Module):
    def __init__(self, layers: int = 5):
        super().__init__()
        self.ssl = SSLModel(layers)
        self.aasist = AASIST(feat_dim=self.ssl.out_dim)

    def forward(self, wav, return_hidden=False):
        feats = self.ssl(wav)                        # (B,T',D)
        logits, hidden = self.aasist(feats, return_hidden=True)
        return (hidden, logits) if return_hidden else logits




# ---------------- load frontend (.safetensors) ----------------



# ---------------- main ----------------
if __name__ == "__main__":
   # ← 改成你的权重路径
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Wav2Vec2_AASIST().to(device)
    # 输出model的总参数量
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # quick sanity-check
    dummy = torch.randn(2, 64600, device=device)   # 2-sec @16k
    hid, logit = model(dummy, return_hidden=True)
    print("sanity logits:", logit.shape)
