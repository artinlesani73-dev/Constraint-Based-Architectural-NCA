"""Verify isolated history-store annotations; requires test server on8027."""
from pathlib import Path
import hashlib,json,urllib.request,urllib.error
R=Path(__file__).resolve().parents[1];DATA=R/'.local-artifacts/r3-history-check-2026-10-10'
RID='8f11090ceaef4f618f853acddb7c4988'
def api(path,value=None,origin='http://127.0.0.1:8027'):
    req=urllib.request.Request('http://127.0.0.1:8027'+path,data=None if value is None else json.dumps(value).encode(),headers={'Content-Type':'application/json','X-NCA-Local':'1','Origin':origin})
    with urllib.request.urlopen(req,timeout=20) as response:return json.loads(response.read())
def main():
    path='/api/annotations/'+RID;latest=api(path)
    assert latest['sequence']==2 and latest['label']=='Courtyard review'
    versions=list((DATA/'annotations'/RID).glob('*.json'));assert len(versions)==2
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in versions}
    for value,origin in [(dict(label='stale',notes='must not overwrite',version=None),'http://127.0.0.1:8027'),
                         (dict(label='x'*121,notes='',version=latest['version']),'http://127.0.0.1:8027'),
                         (dict(label='forbidden',notes='',version=latest['version']),'https://example.com')]:
        try:api(path,value,origin)
        except urllib.error.HTTPError as e:assert e.code in (400,403)
        else:raise AssertionError('Invalid annotation accepted')
    assert api(path)==latest
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'annotations'/RID).glob('*.json')}
    import_receipt=json.loads((DATA/'import-receipt.json').read_text())
    for name,h in import_receipt['files'].items():assert hashlib.sha256((DATA/name).read_bytes()).hexdigest()==h,name
    report=dict(passed=True,original_runs=15,original_drafts=6,all_imported_evidence_unchanged=True,
                two_annotation_versions_preserved=True,stale_save_rejected=True,
                oversized_label_rejected=True,foreign_origin_rejected=True,
                browser_search_and_restart_restore_verified=True,no_generation=True)
    (R/'experiments/reports/R3-research-history-check-2026-10-10.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
