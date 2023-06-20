import re

import pydnsbl

from urllib.parse import urlparse

from features.core import BaseFeature, FeatureType, FeatureClass


class FR01(BaseFeature):
    DESCRIPTION = "Real sender's domain is blacklisted by DNSBL"
    BLOCKED_BY = []
    CLASS = FeatureClass.REPUTATIONAL

    def init(self):
        self.domain_checker = pydnsbl.DNSBLDomainChecker()

    def check(self):
        target = self.domain_checker.check(self.envelope.real_sender.domain)

        return target.blacklisted


class FR02(BaseFeature):
    DESCRIPTION = "Sender's IP address is blacklisted by DNSBL"
    BLOCKED_BY = []
    CLASS = FeatureClass.REPUTATIONAL

    def init(self):
        self.ip_checker = pydnsbl.DNSBLIpChecker()

    def check(self):
        target = self.ip_checker.check(self.envelope.real_sender.ip_address)

        return target.blacklisted


class FR03(BaseFeature):
    DESCRIPTION = "Sender's domain (From header) is blacklisted by DNSBL"
    BLOCKED_BY = []
    CLASS = FeatureClass.REPUTATIONAL

    def init(self):
        self.domain_checker = pydnsbl.DNSBLDomainChecker()
        self.url_rxp = re.compile(r'http?://\S+|https?://\S+|www\.\S+')

    def check(self):
        target = False
        urls = self.url_rxp.findall(str(self.envelope.text))

        for url in urls.group():
            domain = urlparse(url).netloc
            target |= self.domain_checker.check(domain).blacklisted

        return target
