import librosa
import torch

from transducer.models.parakeet import Parakeet
from safetensors.torch import save_file
from huggingface_hub import login, upload_folder, HfApi
import os



def main() -> int:
    audio_path = "/home/ubuntu/transducer/fugitivepieces_02_pope_64kb.mp3"
    audio, _sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Parakeet.from_pretrained()
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model._greedy_decode(audio_tensor.to(device))

    transcript = output.labels[0] if output.labels else ""
    print(transcript)


    return 0



if __name__ == "__main__":
    raise SystemExit(main())
