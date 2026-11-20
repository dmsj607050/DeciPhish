from ..util import digest


def evidence(claim, value, kind, ref_id, locator, observed_at, extractor_version,
             limitations, verification='UNVERIFIED'):
    """Build a deterministic, schema-compatible evidence record."""
    if kind not in ('HTML', 'URL', 'SCREENSHOT'):
        raise ValueError('UNSUPPORTED_EVIDENCE_SOURCE')
    if not all(isinstance(item, str) and item for item in
               (claim, ref_id, locator, observed_at, extractor_version)):
        raise ValueError('INVALID_EVIDENCE_METADATA')
    if verification not in ('UNVERIFIED', 'DERIVED', 'INDEPENDENTLY_VERIFIED'):
        raise ValueError('INVALID_EVIDENCE_VERIFICATION')
    limits = sorted(set(limitations))
    if any(not isinstance(item, str) or not item for item in limits):
        raise ValueError('INVALID_EVIDENCE_LIMITATION')

    source = {'kind': kind, 'ref_id': ref_id, 'locator': locator}
    identity = [claim, value, source, observed_at, extractor_version]
    return {
        'evidence_id': 'ev-' + digest(identity)[:32],
        'claim': claim,
        'value': value,
        'source': source,
        'observed_at': observed_at,
        'verification': verification,
        'extractor_version': extractor_version,
        'limitations': limits,
    }
