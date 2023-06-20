from features.reputational import FR01, FR02, FR03
from features.ml import FML01, FML02
from features.header import (
    FH01, FH02, FH03, FH04, FH05, FH06, FH07, FH08, FH09,
    FH10, FH11, FH12, FH13, FH14, FH15, FH16, FH17, FH18,
    FH19, FH20, FH21, FH22, FH23, FH24
)


class ClassifierVerdict:
    BENIGN = "Benign"
    SPAM = "Spam"
    MALICIOUS = "Malicious"


class BaseClassifierRule:
    DESCRIPTION = None
    VERDICT = None

    def check(self):
        raise NotImplementedError


class CR01(BaseClassifierRule):
    DESCRIPTION = "Both machine learning engines classifed email as spam"
    VERDICT = ClassifierVerdict.SPAM

    def check(self):
        result = set([FML01, FML02]).issubset(self.features)

        return result


class CR02(BaseClassifierRule):
    DESCRIPTION = "Both SPF and DKIM checks failed"
    VERDICT = ClassifierVerdict.SPAM

    def check(self):
        result = set([FH12, FH13]).issubset(self.features)

        return result


class CR03(BaseClassifierRule):
    DESCRIPTION = "Real sender's domain is blacklisted by DNSBL"
    VERDICT = ClassifierVerdict.SPAM

    def check(self):
        result = set([FR01]).issubset(self.features)

        return result


class CR04(BaseClassifierRule):
    DESCRIPTION = "Sender's IP address is blacklisted by DNSBL"
    VERDICT = ClassifierVerdict.SPAM

    def check(self):
        result = set([FR02]).issubset(self.features)

        return result


class CR05(BaseClassifierRule):
    DESCRIPTION = "Sender's domain (From header) is blacklisted by DNSBL"
    VERDICT = ClassifierVerdict.SPAM

    def check(self):
        result = set([FR03]).issubset(self.features)

        return result


class CR06(BaseClassifierRule):
    DESCRIPTION = "Known threat artor pattern"
    VERDICT = ClassifierVerdict.MALICIOUS

    def check(self):
        result = set([FH14, FH18]).issubset(self.features)

        return result
