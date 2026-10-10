"""Append-only researcher annotations, outside immutable run directories."""
import datetime,json,threading,uuid
guard=threading.Lock()
def latest(data,rid):
    records=[json.loads(p.read_text(encoding='utf-8')) for p in (data/'annotations'/rid).glob('*.json')]
    return max(records,key=lambda v:v['sequence']) if records else dict(version=None,sequence=0,label='',notes='')
def save(data,rid,value):
    if len(rid)!=32 or any(c not in '0123456789abcdef' for c in rid) or not (data/'runs'/rid/'request.json').is_file():raise ValueError('Unknown run')
    if set(value)!={'label','notes','version'} or not isinstance(value['label'],str) or not isinstance(value['notes'],str) or len(value['label'])>120 or len(value['notes'])>8000:raise ValueError('Label limit120; notes limit8000 characters')
    with guard:
        previous=latest(data,rid)
        if value['version']!=previous['version']:raise ValueError('Notes changed in another tab. Reopen the run before saving.')
        record=dict(value,version=uuid.uuid4().hex,sequence=previous['sequence']+1,run_id=rid,saved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        folder=data/'annotations'/rid;folder.mkdir(parents=True,exist_ok=True)
        temp=folder/(record['version']+'.tmp');temp.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');temp.rename(temp.with_suffix('.json'))
        return record
