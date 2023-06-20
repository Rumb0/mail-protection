import aiohttp
import asyncio
import json

from contextlib import suppress


class ClassifierVerdict:
    BENIGN = "Benign"
    SPAM = "Spam"
    MALICIOUS = "Malicious"


class Classifier:
    def __init__(self, analyzers: list, rulelist: list):
        self.analyzers = analyzers
        self.crules = rulelist

    async def run(self, mailbuffer: asyncio.Queue, cid: str, controller_url: str):
        try:
            while True:
                email = await mailbuffer.get()
                email["features"] = []

                verdict, _ = await self.verdict(email)

                data = dict(
                    cid = cid,
                    mid = email["id"],
                    verdict = verdict,
                )

                async with aiohttp.ClientSession() as session:
                    async with session.post(controller_url + '/submit', json=data) as resp:
                        if not resp.ok:
                            print("Cannot submit verdict!")

        except asyncio.CancelledError:
            pass


    async def verdict(self, content) -> str:
        verdict = ClassifierVerdict.BENIGN
        rules_triggered = set()

        content["features"] = set()

        # There we gather all features
        for analyzer in self.analyzers:
            result = await analyzer.analyze_sequential(content)
            content["features"] = content["features"].union(result.features)


        # And then apply rules
        for rule in self.crules:
            rule.features = content["features"]

            if rule.check(rule):
                if verdict == ClassifierVerdict.BENIGN:
                    verdict = rule.VERDICT
                elif verdict == ClassifierVerdict.SPAM and rule.VERDICT == ClassifierVerdict.MALICIOUS:
                    verdict = rule.VERDICT
                else:
                    pass

                rules_triggered.add(rule)

        return verdict, rules_triggered
