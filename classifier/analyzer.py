from features.core import BaseFeature

class AnalysisResult:

    def __init__(self):
        self.features_checked = set()
        self.features = set()

        self.verdict_capable = False
        self.verdict = None


    def setfeaturelist(self, fealurelist: list):
        self.features = set(fealurelist)

    def addfeature(self, feature: BaseFeature):

        # Set obj ensures there is no dublicates
        self.features_checked.add(feature)


class Analyzer:
    def __init__(self, features: list):
        self.features = set(features)

        for feature in self.features:
            if hasattr(feature, 'init'):
                feature.init(feature)


    async def analyze_sequential(self, envelope, analysis_done_cb=None) -> AnalysisResult:
        self.envelope = envelope
        target = AnalysisResult()
        target.setfeaturelist(self.features)

        for feature in self.features:
            feature.envelope = envelope
            feature.features = target.features

            if feature.check(feature):
                target.addfeature(feature)

        if analysis_done_cb is not None:
            target = await analysis_done_cb(target)

        return(target)

