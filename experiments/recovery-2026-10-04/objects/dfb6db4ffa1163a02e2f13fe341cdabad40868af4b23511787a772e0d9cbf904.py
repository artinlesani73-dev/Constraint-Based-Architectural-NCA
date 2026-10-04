from pathlib import Path
from PIL import Image
p=Path('C:/Users/artin/Documents/Codex/outputs/G9-Final-Review-2026-10-04')
for steps in [64,128]:
 files=sorted((p/f'visuals-final-{steps}').glob('*.png'))
 for i in range(0,len(files),4):
  im=Image.new('RGB',(1440,1020),'white')
  for j,f in enumerate(files[i:i+4]):im.paste(Image.open(f).resize((720,510)),((j%2)*720,(j//2)*510))
  im.save(p/f'overview-{steps}-{i//4}.png')
