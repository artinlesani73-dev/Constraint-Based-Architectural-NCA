from pathlib import Path
p=Path('render_g6_reserved.py').read_text().replace('G6-Reserved-Review-2026-10-04','G7-Final-Review-2026-10-04').replace("OUT/'split-manifest.json'","OUT/'scene-index.json'").replace('G6 frozen evaluation','G7 frozen evaluation').replace("replace('g1-','')","replace('g1-','').replace('g7-','')").replace("print('Saved four geometry plates.')","print('Saved eleven geometry plates.')")
Path('render_g7.py').write_text(p)
