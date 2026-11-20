"""Bounded lexical hints from inline JS. No execution or dataflow claims.

Only exact call shapes and simple literal bindings are recognized. Comments and
string contents are never treated as executable code. Dynamic destinations and
incomplete scans stay explicitly unknown. No external source is fetched.
"""
import re
from urllib.parse import urljoin

from ..util import redact_url

IDENT = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')


def tokens(source, max_chars=100_000, max_tokens=12_000):
    result=[];i=0;limit=min(len(source),max_chars);partial=len(source)>limit
    while i<limit and len(result)<max_tokens:
        c=source[i]
        if c.isspace():i+=1;continue
        if source.startswith('//',i):
            end=source.find('\n',i+2,limit);i=limit if end<0 else end+1;continue
        if source.startswith('/*',i):
            end=source.find('*/',i+2,limit)
            if end<0:partial=True;break
            i=end+2;continue
        if c=='/' and (not result or result[-1][1] in ('=', '(', ':', ',', 'return', '!', '?', '[', '{', ';')):
            # Regex literals are data too; do not mistake example call text
            # inside /.../ for an invocation.
            i+=1;in_class=False;closed=False
            while i<limit:
                if source[i]=='\\':i+=2;continue
                if source[i]=='[':in_class=True
                elif source[i]==']':in_class=False
                elif source[i]=='/' and not in_class:i+=1;closed=True;break
                elif source[i]=='\n':break
                i+=1
            if not closed:partial=True;break
            while i<limit and source[i].isalpha():i+=1
            result.append(('opaque','<regex>'));continue
        if c in ('"',"'",'`'):
            quote=c;start=i;i+=1;value=[];dynamic=False;closed=False
            while i<limit:
                c=source[i]
                if c==quote:i+=1;closed=True;break
                if c=='\\' and i+1<limit:
                    nxt=source[i+1]
                    value.append({'n':'\n','t':'\t','r':'\r'}.get(nxt,nxt));i+=2;continue
                if quote=='`' and source.startswith('${',i):dynamic=True
                if len(value)<2048:value.append(c)
                i+=1
            if not closed:partial=True;break
            # Escaped hex/unicode and interpolated templates are not resolved.
            raw=source[start:i]
            unsupported=bool(re.search(r'\\[xu]',raw)) or dynamic or i-start>2050
            result.append(('dynamic' if unsupported else 'str',''.join(value)))
            continue
        match=IDENT.match(source,i,limit)
        if match:result.append(('id',match.group()));i=match.end();continue
        result.append(('p',c));i+=1
    return result,partial or i<limit


def analyze_script(source, base):
    ts,partial=tokens(source);values=[t[1] if t[0]!='str' else '<literal>' for t in ts]
    bindings={};objects={};selectors={};reads=[];requests=[];redirects=[];handlers=[];assignments=[];prevent=False;unknown=0

    def literal(index):
        if index>=len(ts):return None
        kind,value=ts[index]
        if kind=='str':return value
        if kind=='id':return bindings.get(value)
        return None

    def argument(index):
        value=literal(index)
        # Refuse to resolve the first fragment of a concatenation/expression.
        return value if index+1<len(ts) and values[index+1] in (',',')',';','}') else None

    def target(value):
        if value is None:return 'DYNAMIC_UNKNOWN'
        return redact_url(urljoin(base,value))[:240]

    def options(index):
        if index>=len(ts):return {}
        if ts[index][0]=='id':return objects.get(values[index],{})
        if values[index]!='{':return {}
        result={};depth=0;end=None
        for j in range(index,min(index+512,len(ts))):
            if values[j] in ('{','[','('):depth+=1
            elif values[j] in ('}',']',')'):
                depth-=1
                if depth==0:end=j;break
            if depth==1 and j+2<len(ts) and ts[j][1] in ('url','type','method') and values[j+1]==':':
                # Full simple literal or previously bound string, never a
                # concatenation, nested payload property or executable value.
                if j+3<len(ts) and values[j+3] in (',','}'):
                    result[ts[j][1]]=literal(j+2)
                else:
                    # Resolve only a bounded concatenation of known strings.
                    # No eval, interpolation, call, unknown variable or payload.
                    result[ts[j][1]]=None;parts=[];cursor=j+2
                    for _ in range(8):
                        value=literal(cursor)
                        if value is None:break
                        parts.append(value);cursor+=1
                        if cursor<len(ts) and values[cursor] in (',','}'):
                            result[ts[j][1]]=''.join(parts) if sum(map(len,parts))<=2048 else None;break
                        if cursor>=len(ts) or values[cursor]!='+':break
                        cursor+=1
                    else:result[ts[j][1]]=None
        return result if end is not None else {}

    for i,(kind,name) in enumerate(ts):
        if kind!='id':continue
        if name=='function':objects.clear()
        if i+2<len(ts) and values[i+1] in ('+','-','*','/') and values[i+2] in ('=','+','-'):
            bindings.pop(name,None);objects.pop(name,None);selectors.pop(name,None)
        # Resolve constants only. Reassignments invalidate old bindings.
        if i+1<len(ts) and values[i+1]=='=' and (i==0 or values[i-1]!='.'):
            bindings.pop(name,None);objects.pop(name,None);selectors.pop(name,None)
            v=literal(i+2)
            if i+2<len(ts) and ts[i+2][0]=='id':objects.pop(values[i+2],None)
            if v is not None and i+3<len(ts) and values[i+3] in (';',',',')'):
                bindings[name]=v
            if i+2<len(ts) and values[i+2]=='{':objects[name]=options(i+2)
        if i>=2 and values[i-1]=='.' and values[i-2] in objects:
            if values[i+1:i+2]==['='] or values[i+1:i+3] in (['+','='],['-','=']):
                # A mutated object is no longer a known initial configuration.
                objects.pop(values[i-2],None)
        if name in ('querySelector','getElementById') and i>0 and values[i-1]=='.' and values[i+1:i+2]==['(']:
            s=literal(i+2)
            if s is None or len(s)>100:continue
            if name=='getElementById':s='#'+s
            if values[i+3:i+6]==[')','.','value']:
                reads.append(s)
            if i>=4 and values[i-3]=='=' and ts[i-4][0]=='id':selectors[values[i-4]]=s
        if name=='value' and i>=2 and values[i-1]=='.' and values[i-2] in selectors:
            reads.append(selectors[values[i-2]])
        if name=='addEventListener' and i>=2 and values[i-1]=='.' and values[i+1:i+2]==['('] and literal(i+2)=='submit':
            handlers.append(selectors.get(values[i-2],'DYNAMIC_UNKNOWN'))
        if name=='preventDefault' and values[i+1:i+3]==['(',')']:prevent=True
        if name=='val' and values[i+1:i+3]==['(',')'] and i>=5 and values[i-4]=='(' and values[i-2:i]==[')','.'] and values[i-5] in ('$','jQuery'):
            selector=literal(i-3)
            if selector and len(selector)<=100:reads.append(selector)
        method=None;index=None;api=None
        if name in ('ajax','post','get','getJSON') and i>=2 and values[i-2] in ('$','jQuery') and values[i-1:i+2]==['.',name,'(']:
            api='jquery.'+name
            if name=='ajax':
                opts=options(i+2);v=opts.get('url');method=opts.get('method') or opts.get('type') or 'UNSPECIFIED'
                method=method.upper() if isinstance(method,str) else 'UNSPECIFIED'
                row=dict(api=api,method=method,target=target(v));unknown+=v is None
                if row not in requests:requests.append(row)
            else:method='POST' if name=='post' else 'GET';index=i+2
        elif name in ('post','get','put','patch','delete') and i>=2 and values[i-2:i]==['axios','.'] and values[i+1:i+2]==['(']:
            method=name.upper();index=i+2;api='axios'
        elif name=='fetch' and values[i+1:i+2]==['('] and (i==0 or values[i-1] not in ('.','function') or
                (i>=2 and values[i-2] in ('window','self','globalThis'))):
            method='UNSPECIFIED';index=i+2;api='fetch'
            # Only an options object's top-level property belongs to this
            # request. Do not scan into a following call or nested body data.
            depth=0;paren=1
            for j in range(i+3,min(i+128,len(ts)-2)):
                if values[j]=='(':paren+=1
                elif values[j]==')':
                    paren-=1
                    if paren==0:break
                if values[j]=='{':depth+=1
                elif values[j]=='}':depth-=1
                if depth==1 and paren==1 and ts[j][1]=='method' and values[j+1]==':' and ts[j+2][0]=='str':
                    method=ts[j+2][1].upper();break
        elif name=='open' and i>=2 and values[i-1]=='.' and values[i+1:i+2]==['('] and literal(i+2) in ('GET','POST','PUT','PATCH','DELETE') and values[i+3:i+4]==[',']:
            # Receiver is not proven XMLHttpRequest. Preserve the uncertainty.
            method=literal(i+2);index=i+4;api='open_receiver_unverified'
        if index is not None:
            v=argument(index);unknown+=v is None
            row=dict(api=api,method=method,target=target(v))
            if row not in requests:requests.append(row)
        if name=='location':
            index=None
            if values[i+1:i+3]==['.','href'] and values[i+3:i+4]==['=']:index=i+4
            elif values[i+1:i+2]==['=']:index=i+2
            elif values[i+1:i+3] in (['.','assign'],['.','replace']) and values[i+3:i+4]==['(']:index=i+4
            if index is not None:
                v=argument(index);unknown+=v is None
                if target(v) not in redirects:redirects.append(target(v))
        if name in ('action','method') and i>=2 and values[i-1]=='.' and values[i-2] in selectors and values[i+1:i+2]==['=']:
            v=argument(i+2)
            assignments.append(dict(form=selectors[values[i-2]],property=name,
                value=target(v) if name=='action' else (v.upper() if v else 'DYNAMIC_UNKNOWN')))
    if not (reads or handlers or requests or redirects or assignments):return None,partial
    result=dict(analysis='STATIC_COOCCURRENCE_NOT_DATAFLOW_OR_EXECUTION')
    if handlers:result['submit_targets']=list(dict.fromkeys(handlers))[:3]
    if prevent:result['prevents_default_present']=True
    if reads:result['input_reads']=list(dict.fromkeys(reads))[:4]
    if requests:result['requests']=requests[:4]
    if redirects:result['redirects']=redirects[:3]
    if assignments:result['form_assignments']=assignments[:3]
    if unknown:result['unresolved_targets']=unknown
    if partial or len(requests)>4 or len(reads)>4 or len(redirects)>3:
        result['partial']=True;partial=True
    return result,partial
