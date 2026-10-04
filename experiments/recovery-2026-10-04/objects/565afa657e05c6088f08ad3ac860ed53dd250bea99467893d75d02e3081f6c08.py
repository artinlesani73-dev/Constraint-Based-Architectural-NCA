from pathlib import Path
s=Path('render_g7.py').read_text().replace('G7-Final-Review-2026-10-04','G8-Final-Review-2026-10-04-v2').replace('G7 frozen evaluation','G8 frozen evaluation').replace("replace('g7-','')","replace('g7-','').replace('g8-','')").replace('eleven geometry plates','fifteen geometry plates')
Path('render_g8.py').write_text(s)
s=Path('audit_g7_results.py').read_text().replace("OUT=Path('C:/Users/artin/Documents/Codex/outputs/G7-Final-Review-2026-10-04')","OUT=Path('C:/Users/artin/Documents/Codex/outputs/G8-Final-Review-2026-10-04-v2')")
a=s.index("for folder in [");b=s.index('comparison=[]',a)
s=s[:a]+"old=json.loads((BASE/'G7-Final-Review-2026-10-04/result.json').read_text())['observations']\n"+s[b:]
s=s.replace('g6_pass','g7_pass').replace('g7_pass=o','g8_pass=o').replace('g6_failed_families','g7_failed_families').replace("g7_failed_families=[k for k,v in o","g8_failed_families=[k for k,v in o").replace("c['g7_pass']!=c['g7_pass']","c['g7_pass']!=c['g8_pass']").replace('G6 has not been evaluated on the fresh G7 reserved cohort','G7 has not been evaluated on the fresh G8 reserved cohort')
Path('audit_g8_results.py').write_text(s)
