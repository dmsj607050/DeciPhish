"""Offline task-specific DOM reduction; never execute scripts or infer a label."""
from collections import Counter
from importlib.metadata import version
import re
from urllib.parse import urljoin, urlsplit

from lxml import html, etree

from ..util import canonical, sanitize, redact_url
from .script_hints import analyze_script


INTERACTION = re.compile(r'log.?in|sign.?in|account|password|verify|verification|auth|payment|checkout|'
                         r'entrar|acesso|senha|connexion|anmelden|continuar|continue|join|view|usuario|documento|'
                         r'suspended|disabled|warning|invited|reported|abuse|登录|登陆|验证|密码|支付|继续|停用|封禁', re.I)
IDENTITY = re.compile(r'copyright|©|official|privacy|terms|contact|about|legal|官方|隐私|版权|联系', re.I)
BOILERPLATE = re.compile(r'(^|[-_\s])(nav|menu|footer|cookie|breadcrumb|social|share)([-_\s]|$)', re.I)


def extract_html(case, store, now, max_chars=4000, max_items=80, max_text_segments=12,
                 max_anchor_links=10, max_resource_links=4):
    from . import evidence
    resource = case['html_resource']
    data = store.read(case['case_id'], resource)
    # libxml otherwise uses a legacy encoding on UTF-8 pages lacking charset.
    try:
        data.decode('utf-8')
        encoding = 'utf-8'
    except UnicodeDecodeError:
        encoding = None
    parser = html.HTMLParser(no_network=True, recover=True, remove_comments=True, encoding=encoding)
    try:
        dom = html.fromstring(data, parser=parser)
    except etree.ParserError:
        raise ValueError('HTML_PARSE_FAILED') from None
    tree = dom.getroottree()
    provider = 'rules_lxml@' + version('lxml') + '+task-dom-v4-leaf-interaction'
    base = case['observed_url'] or ''
    items, pending, notes, omitted = [], [], set(), Counter()
    used = 0

    def text(node, limit=240):
        parts=[];remaining=max(4000,limit*4)
        for chunk in node.itertext():
            parts.append(chunk[:remaining]);remaining-=min(len(chunk),remaining)
            if remaining<=0:break
        return ' '.join(' '.join(parts).split())[:limit]

    def resolve(value):
        return redact_url(urljoin(base, value)) if base else redact_url(value)

    def add(claim, value, node, locator=None):
        value = sanitize(value)
        pending.append((claim,value,locator or tree.getpath(node)))
        return True

    def purpose(node):
        label=node.get('aria-label') or node.get('data-name') or node.get('name') or node.get('id') or ''
        label=' '.join(label.split())[:100]
        hint='UNKNOWN'
        for pattern,name in [(r'trial|demo|试用','TRIAL'),(r'search|buscar|搜索','SEARCH'),
                (r'login|log.?in|sign.?in|登录','LOGIN'),(r'register|sign.?up|注册','REGISTRATION'),
                (r'contact|联系','CONTACT'),(r'checkout|payment|支付','PAYMENT')]:
            if re.search(pattern,label,re.I):hint=name;break
        return label,hint

    def visibility_hint(node):
        for parent in (node,*node.iterancestors()):
            if 'hidden' in parent.attrib or parent.get('aria-hidden')=='true' or re.search(
                    r'display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)',parent.get('style',''),re.I):
                return 'STATIC_HIDDEN_OR_TRANSPARENT_HINT'
        return 'UNKNOWN'

    def boilerplate(node):
        return any(parent.tag in ('nav', 'footer', 'aside') or parent.get('role')=='navigation' or
            BOILERPLATE.search((parent.get('class') or '')+' '+(parent.get('id') or ''))
            for parent in (node, *node.iterancestors()))

    resources=[]
    for node in dom.xpath('//script[@src]|//img[@src]|//link[@href]'):
        attr='href' if node.tag=='link' else 'src'
        resources.append((node.tag,attr,node.get(attr),tree.getpath(node),node.get('alt',''),node.get('rel','')))
    behavior=[];scanned=0;script_chars=0
    for node in dom.xpath('//script'):
        if node.get('src'):
            notes.add('html:EXTERNAL_SCRIPTS_NOT_FETCHED');continue
        kind=node.get('type','').lower()
        if kind not in ('','module','text/javascript','application/javascript'):continue
        if scanned>=24 or script_chars>=300_000:
            notes.add('html:STATIC_SCRIPT_SCAN_PARTIAL');break
        source=node.text or '';allowed=min(100_000,300_000-script_chars)
        summary,partial=analyze_script(source[:allowed],base)
        script_chars+=min(len(source),allowed);scanned+=1
        if partial or len(source)>allowed:notes.add('html:STATIC_SCRIPT_SCAN_PARTIAL')
        if summary:
            if len(behavior)<3:behavior.append((summary,tree.getpath(node)))
            else:notes.add('html:STATIC_SCRIPT_HINTS_OMITTED')
    for node in dom.xpath('//*[@onsubmit or @onclick or @onchange or @oninput]')[:16]:
        for attr in ('onsubmit','onclick','onchange','oninput'):
            source=node.get(attr)
            if source is None:continue
            summary,partial=analyze_script(source[:4000],base)
            if partial or len(source)>4000:notes.add('html:STATIC_SCRIPT_SCAN_PARTIAL')
            if summary:
                summary['event_attribute']=attr
                if len(behavior)<3:behavior.append((summary,tree.getpath(node)+'/@'+attr))
                else:notes.add('html:STATIC_SCRIPT_HINTS_OMITTED')
    if scanned:notes.add('html:STATIC_JS_LIMITED_PATTERNS_ONLY')
    for node in dom.xpath('//script|//style|//template'):
        node.drop_tree()
    for node in dom.xpath('//textarea'):
        node.text=None
        for child in list(node):
            node.remove(child)
    prefilled={node for node in dom.xpath('//input[@value]')}
    # Only static submit/button captions, never text/password/hidden values.
    captions={node:node.get('value','') for node in dom.xpath('//input[@value]')
              if node.get('type','text').lower() in ('submit','button','reset')}
    for node in dom.xpath('//input'):
        node.attrib.pop('value',None)

    title=dom.find('.//title')
    if title is not None:
        add('html.document_title',{'text':text(title,240)},title)
    for node in dom.xpath('//meta[translate(@name,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="description"]')[:1]:
        add('html.meta_description',{'text':' '.join(node.get('content','').split())[:240],'name':'description'},node)
    for summary,locator in behavior:
        add('html.script_behavior',summary,dom,locator)

    # Form association is retained; equal controls in distinct forms are
    # distinct evidence. Hidden fields retain presence, never their values.
    forms=dom.xpath('//form')
    forms.sort(key=lambda n:not bool(n.xpath('.//input[translate(@type,"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz")="password"]')))
    for node in forms:
        raw=node.get('action')
        label,hint=purpose(node)
        value={'method':node.get('method','get').upper(),
            'method_source':'DECLARED' if node.get('method') is not None else 'HTML_DEFAULT',
            'action_raw':redact_url(raw) if raw is not None else None,
            'action_resolved':resolve(raw or '') if base else None,
            'action_source':'MISSING_DEFAULT' if raw is None else 'EMPTY_DEFAULT' if not raw else 'DECLARED',
            'purpose_hint':hint,'visibility_hint':visibility_hint(node)}
        if label:value['label_hint']=label
        if node.get('onsubmit') is not None:value['inline_submit_handler_present']=True
        add('html.form',value,node)
    fields=dom.xpath('//input|//textarea|//select|//button|//iframe')
    fields.sort(key=lambda n:0 if n.tag=='input' and n.get('type','text').lower()=='password' else
        1 if n.tag=='input' and n.get('type','text').lower() in ('email','tel') else
        5 if n.tag=='input' and n.get('type','text').lower()=='hidden' else
        {'iframe':2,'textarea':3,'select':3,'input':3,'button':4}[n.tag])
    seen_buttons=set()
    for node in fields:
        if node.tag in ('input','textarea','select'):
            value={'type':node.get('type','text').lower() if node.tag=='input' else node.tag,'name_present':bool(node.get('name')),
                'autocomplete_hint':node.get('autocomplete'), 'prefilled_value_present':node in prefilled,
                'hidden_attribute':'hidden' in node.attrib}
            form=next(node.iterancestors('form'),None)
            explicit=node.get('form')
            if explicit:
                matches=dom.xpath('//form[@id=$target]',target=explicit)
                form=matches[0] if matches else None
            if form is not None:
                value['form_locator']=tree.getpath(form)
            hint=node.get('aria-label') or node.get('placeholder')
            if not hint and node.get('id'):
                labels=dom.xpath('//label[@for=$target]',target=node.get('id'))
                hint=' '.join(text(label,100) for label in labels[:2])
            if hint:
                value['label_hint']=' '.join(hint.split())[:100]
            elif node in captions:
                value['label_hint']=' '.join(captions[node].split())[:100]
            if node.get('formaction') is not None:
                value['formaction_resolved']=resolve(node.get('formaction'))
            add('html.input',value,node)
        elif node.tag=='button':
            value={'type':node.get('type','submit'),'text':text(node,100) or
                   ' '.join(node.xpath('.//img/@alt'))[:100] or node.get('aria-label','')[:100]}
            form=next(node.iterancestors('form'),None)
            if form is not None:
                value['form_locator']=tree.getpath(form)
            if node.get('formaction') is not None:
                value['formaction_resolved']=resolve(node.get('formaction'))
            if node.get('formmethod'):
                value['formmethod']=node.get('formmethod').upper()
            key=canonical(value)
            if key in seen_buttons:
                omitted['html.button:duplicate']+=1
                continue
            seen_buttons.add(key)
            add('html.button',value,node)
        else:
            raw=node.get('src')
            add('html.iframe',{'src':resolve(raw) if raw is not None else None,
                'sandbox':node.get('sandbox')},node)

    # Do not delete whole footer/nav subtrees: identity and login evidence
    # may live there. Only repetitive, low-priority text is suppressed.
    candidates=[]
    text_nodes=dom.xpath('//h1|//h2|//h3|//label|//p')
    # Bounded short leaf containers and accessibility/SVG text. Avoid whole
    # application subtrees, descendants of existing paragraphs and user values.
    extra=dom.xpath('//div|//span|//strong|//b|//font|//td|//text|//*[@role="heading"]')
    if len(extra)>2000:notes.add('html:LEAF_TEXT_SCAN_PARTIAL')
    for node in extra[:2000]:
        if node in text_nodes or node.xpath('ancestor::p|ancestor::label|ancestor::button|ancestor::textarea|ancestor::select|ancestor::h1|ancestor::h2|ancestor::h3'):continue
        if len(node)>12 or node.xpath('.//div|.//span|.//p|.//input|.//textarea|.//select|.//button'):continue
        value=text(node,241)
        if not value or len(value)>240:continue
        text_nodes.append(node)
    for node in text_nodes:
        value=text(node)
        if not value or node.xpath('ancestor-or-self::*[@hidden or @aria-hidden="true"]'):
            continue
        in_form=any(True for _ in node.iterancestors('form'))
        rank=0 if node.tag in ('h1','h2','h3','label') or in_form or INTERACTION.search(value) else 1 if IDENTITY.search(value) else 3 if boilerplate(node) else 2
        candidates.append((rank,node,value))
    seen_text=set();retained=0
    for rank,node,value in sorted(candidates,key=lambda row:row[0]):
        key=value.casefold()
        if key in seen_text:
            omitted['html.text_segment:duplicate']+=1
            continue
        seen_text.add(key)
        if retained>=max_text_segments or (rank==3 and retained>=2):
            omitted['html.text_segment:low_priority']+=1
            continue
        if add('html.text_segment',{'text':value,'visibility':'UNKNOWN','role_hint':node.tag},node):
            retained+=1

    anchors=[]
    for node in dom.xpath('//a[@href]'):
        raw=node.get('href','');target=resolve(raw);label=text(node,100)
        if not raw.strip() or raw.startswith(('#','javascript:','data:')):
            omitted['html.anchor:non_navigation']+=1
            continue
        if not label and not node.xpath('.//img[@alt]'):
            omitted['html.anchor:empty']+=1
            continue
        if not label:
            label=' '.join(node.xpath('.//img/@alt'))[:100]
        try:
            outside=bool(base and urlsplit(target).hostname!=urlsplit(base).hostname)
        except ValueError:
            outside=False
        rank=0 if INTERACTION.search(label+' '+raw) else 1 if IDENTITY.search(label) else 2 if outside else 4 if boilerplate(node) else 3
        anchors.append((rank,node,label,target,raw))
    seen_links=set();retained=0
    for _,node,label,target,raw in sorted(anchors,key=lambda row:row[0]):
        key=(label.casefold(),target)
        if key in seen_links:
            omitted['html.anchor:duplicate']+=1
            continue
        seen_links.add(key)
        if retained>=max_anchor_links:
            omitted['html.anchor:limit']+=1
            continue
        if add('html.anchor',{'text':label,'href_raw':redact_url(raw),
                'href_resolved':target if base else None,'rel':node.get('rel')},node):
            retained+=1

    interactions=dom.xpath('//*[@role="button" or @role="link" or @onclick]')
    if len(interactions)>16:notes.add('html:INTERACTION_SCAN_PARTIAL')
    for node in interactions[:16]:
        if node.tag in ('button','input','a'):continue
        label=node.get('aria-label') or text(node,100) or node.get('title') or ''
        if not label or visibility_hint(node)!='UNKNOWN':continue
        add('html.interaction',dict(text=label[:100],tag=node.tag,role=node.get('role'),
            inline_handler_present=node.get('onclick') is not None,
            target='DYNAMIC_UNKNOWN' if node.get('onclick') else 'NOT_DECLARED'),node)

    # Preserve meaningful accessible image identity text independently of
    # resource-origin limits. Generic placeholders and long prose are excluded.
    image_hints=0
    for node in dom.xpath('//img[@alt or @aria-label or @title]'):
        label=' '.join((node.get('aria-label') or node.get('alt') or node.get('title') or '').split())
        if not 2<=len(label)<=100 or not re.search(r'[^\W\d_]',label) or label.casefold() in ('image','img','logo','icon','...'):continue
        add('html.image_text',dict(text=label,role_hint='IMAGE_TEXT_NOT_VERIFIED_BRAND'),node)
        image_hints+=1
        if image_hints>=3:break

    # Diverse resource origins and image identity hints, not every bundle.
    resources.sort(key=lambda row:0 if 'canonical' in row[5].split() else 1 if row[0]=='img' and row[4] else 2 if row[0]=='img' else 3)
    seen_resources=set();retained=0
    for tag,attr,uri,locator,alt,rel in resources:
        target=resolve(uri)
        try:
            host=urlsplit(target).hostname
        except ValueError:
            host=None
        key=(tag,host) if tag=='script' or (tag=='link' and 'canonical' not in rel.split()) else (tag,target)
        if key in seen_resources:
            omitted['html.resource_link:repeated_host']+=1
            continue
        seen_resources.add(key)
        if retained>=max_resource_links:
            omitted['html.resource_link:limit']+=1
            continue
        value={'tag':tag,'attr':attr,'uri':target}
        if alt:
            value['alt']=' '.join(alt.split())[:100]
        if rel:
            value['rel']=rel[:80]
        if add('html.resource_link',value,dom,locator):
            retained+=1
    # Allocate the unchanged value budget across meaning, behavior and controls
    # before repetitive frames/resources. All returned IDs still refer to full
    # immutable EvidenceItems, never synthesized summaries of unseen content.
    counts=Counter()
    def rank(row):
        index,(claim,value,locator)=row
        position=counts[claim];counts[claim]+=1
        first=claim in ('html.document_title','html.meta_description','html.script_behavior') or (
            claim in ('html.form','html.input','html.text_segment','html.anchor') and position<2) or (
            claim in ('html.button','html.iframe','html.image_text','html.interaction') and position==0)
        priority={'html.document_title':0,'html.meta_description':1,'html.script_behavior':2,
            'html.form':3,'html.input':4,'html.text_segment':5,'html.anchor':6,
            'html.image_text':3,'html.interaction':6,
            'html.button':7,'html.iframe':8,'html.resource_link':9}.get(claim,10)
        return (0 if first else 1,priority,index)
    ranked=[(rank(pair),pair[1]) for pair in enumerate(pending)]
    behavior_chars=0
    for _,(claim,value,locator) in sorted(ranked,key=lambda x:x[0]):
        size=len(canonical(value))
        if used+size>max_chars or len(items)>=max_items or (claim=='html.script_behavior' and behavior_chars+size>1400):
            omitted[claim]+=1;continue
        used+=size
        if claim=='html.script_behavior':behavior_chars+=size
        limits=['PAGE_SELF_REPORT','CSS_VISIBILITY_UNKNOWN']
        if claim=='html.script_behavior':limits+=['STATIC_ANALYSIS_NOT_EXECUTION','COOCCURRENCE_NOT_PROVEN_DATAFLOW','UNSUPPORTED_JS_PATTERNS_MAY_BE_MISSED']
        items.append(evidence(claim,value,'HTML',resource['resource_id'],locator,resource['acquired_at'],provider,limits))
    if omitted:
        notes.update('html:OMITTED:'+claim+':'+str(count) for claim,count in sorted(omitted.items()))
        notes.add('html:PARTIAL_TASK_REDUCTION')
    return items,'PARTIAL' if notes else 'AVAILABLE',sorted(notes),provider
