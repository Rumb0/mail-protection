class FeatureType:
    IMPERSONATION = "Impersonation"
    SPAM = "Spam"
    PHISHING = "Phishing"


class FeatureClass:
    REPUTATIONAL = "Reputational"
    HEURISTIC = "Heuristic"


class BaseFeature:
    BLOCKED_BY = []
    REQUIRES = []
    DESCRIPTION = ""
    CLASS = FeatureClass.HEURISTIC
    TYPE = FeatureType.SPAM

    FMLNAME = None
    FMLVALUE = None

    def __init__(self):
        self.domains = set()
        self.envelope = None
        self.encoded = None

    def __hash__(self):
        return(hash(self.__class__.__name__))

    def __eq__(self, other):
        return(hash(self) == hash(other))

    def encoded(self):
        if FMLNAME is not None:
            return(FMLNAME, FMLVALUE)

    def check(self):
        raise NotImplementedError

    def match(self, envelope):
        raise NotImplementedError


class ComplexFeature(BaseFeature):
    FEATURES = []

    def check(self):
        if all(map(lambda x: x.match(self.envelope), self.FEATURES)):
            return True
        else:
            return False

