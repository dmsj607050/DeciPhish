import hashlib
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit, parse_qsl
import ipaddress

import idna
import publicsuffix2

PSL_PATH = Path(publicsuffix2.__file__).with_name('public_suffix_list.dat')
PSL_SHA = hashlib.sha256(PSL_PATH.read_bytes()).hexdigest()
PSL = publicsuffix2.PublicSuffixList(str(PSL_PATH))


def parse_url(value):
    p = urlsplit(value.strip())
    if not p.hostname or any(c in value for c in '\r\n\t\\'):
        raise ValueError('URL_PARSE_ERROR')
    port = p.port  # Force numeric/range validation.
    try:
        ipaddress.ip_address(p.hostname)
        host, is_ip = p.hostname, True
    except ValueError:
        raw_host = p.hostname.rstrip('.').lower()
        is_ip = False
        try:
            host = idna.encode(raw_host, uts46=True).decode('ascii').lower()
        except idna.IDNAError:
            # Preserve the literal ASCII hostname when a dataset A-label is not
            # valid IDNA (for example an xn-- label that decodes to a disallowed
            # Unicode code point). Keep strict syntax checks and never resolve it.
            labels = raw_host.split('.')
            valid_ascii_host = (raw_host.isascii() and all(
                label and len(label) <= 63 and not label.startswith('-')
                and not label.endswith('-')
                and all(ch.isalnum() or ch == '-' for ch in label)
                for label in labels))
            if not valid_ascii_host:
                raise
            host = raw_host
    registered = None if is_ip else PSL.get_sld(host, strict=True)
    if registered == PSL.get_tld(host, strict=True):
        registered = None
    return p, host, is_ip, registered


def extract_url(case, store, now):
    from . import evidence
    provider = f'rules_urllib+idna@{version("idna")}+psl@{PSL_SHA}'
    try:
        p, host, is_ip, registered = parse_url(case['observed_url'])
    except ValueError as exc:
        if str(exc) != 'URL_PARSE_ERROR':
            raise
        # Keep malformed but observed HTTP(S) URL candidates as raw evidence.
        # Do not normalize, resolve, or request them; mark coverage PARTIAL.
        item = evidence('url.observed_url_raw', case['observed_url'], 'URL', case['case_id'],
                        'observed_url:raw_unparsed', now, provider,
                        ['URL_PARSE_FAILED', 'RAW_OBSERVED_VALUE_PRESERVED', 'UNTRUSTED_DATA'],
                        'UNVERIFIED')
        return [item], 'PARTIAL', ['URL_PARSE_ERROR_PRESERVED_RAW'], provider
    values = {'scheme': p.scheme.lower(), 'hostname_ascii': host,
              'hostname_unicode': host if is_ip else idna.decode(host),
              'registered_domain': registered,
              'subdomain': host[:-len(registered)-1] if registered and host.endswith('.' + registered) else '',
              'port': p.port, 'query_keys': sorted({k for k, _ in parse_qsl(p.query, keep_blank_values=True)}),
              'fragment_present': bool(p.fragment), 'host_is_ip': is_ip,
              'unicode_idn_present': 'xn--' in host, 'username_present': p.username is not None,
              'length': len(case['observed_url'])}
    items = [evidence('url.' + key, value, 'URL', case['case_id'], 'observed_url:' + key,
                      now, provider, ['NOT_IDENTITY_OR_AUTHORIZATION_PROOF'], 'DERIVED')
             for key, value in values.items()]
    notes = [] if p.scheme.lower() in ('http', 'https') else ['url:NETWORK_SCHEME_UNSUPPORTED']
    return items, 'PARTIAL' if notes else 'AVAILABLE', notes, provider
