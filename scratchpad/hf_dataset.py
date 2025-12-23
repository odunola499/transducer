from datasets import load_dataset, Audio
from io import BytesIO
import librosa

data = load_dataset('Jinsaryko/Elise')['train']
data = data.cast_column('audio',Audio(decode = False))

row = data[0]
audio_bytes = BytesIO(row['audio']['bytes'])

array, sr = librosa.load(audio_bytes)
print(array)