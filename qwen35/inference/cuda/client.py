"""Standard-library HTTP client. Supply INF_API_KEY through the environment."""
import argparse,json,os,time,urllib.request,urllib.error
BASE=os.environ.get('SIMPLETOOL_BASE_URL','http://127.0.0.1:8000')
def request(path,body=None,method=None,base=BASE,timeout=180):
    key=os.environ.get('INF_API_KEY')
    headers={'Content-Type':'application/json'}
    if key:headers['Authorization']='Bearer '+key
    data=None if body is None else json.dumps(body,ensure_ascii=False).encode()
    req=urllib.request.Request(base+path,data=data,headers=headers,method=method)
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:status=r.status;result=json.loads(r.read())
    except urllib.error.HTTPError as e:status=e.code;result={'error_body':e.read().decode()}
    return {'http_status':status,'http_wall_ms':(time.perf_counter()-start)*1000,'response':result}
def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['health','ready','config','metrics','list','init','infer','delete'])
    p.add_argument('--json-file');p.add_argument('--scenario-id');p.add_argument('--base',default=BASE)
    content=p.add_mutually_exclusive_group()
    content.add_argument('--include-content',dest='include_content',action='store_true')
    content.add_argument('--no-content',dest='include_content',action='store_false')
    p.set_defaults(include_content=None);a=p.parse_args()
    paths={'health':'/healthz','ready':'/readyz','config':'/v1/config','metrics':'/v1/metrics','list':'/v1/scenarios','init':'/v1/scenarios','infer':'/v1/infer'}
    body=None
    if a.operation in ('init','infer'):
        if not a.json_file:p.error('--json-file is required')
        with open(a.json_file) as f:body=json.load(f)
    if a.include_content is not None:
        if a.operation!='infer':p.error('--include-content/--no-content require infer')
        body['include_content']=a.include_content
    if a.operation=='delete':
        if not a.scenario_id:p.error('--scenario-id is required')
        path='/v1/scenarios/'+a.scenario_id;method='DELETE'
    else:path=paths[a.operation];method=None
    result=request(path,body,method,base=a.base)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if 200<=result['http_status']<300 else 1)
if __name__=='__main__':main()
