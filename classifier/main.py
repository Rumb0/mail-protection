import aiohttp
import asyncio
import json

from contextlib import suppress

from classifier import Classifier
from analyzer import Analyzer
from features.reputational import FR01, FR02, FR03
from features.ml import FML01, FML02
from features.header import (
    FH01, FH02, FH03, FH04, FH05, FH06, FH07, FH08, FH09,
    FH10, FH11, FH12, FH13, FH14, FH15, FH16, FH17, FH18,
    FH19, FH20, FH21, FH22, FH23, FH24
)
from classifier_rules import CR01, CR02, CR03, CR04, CR05, CR06


CLASSIFIER_CONTROLLER_URL = 'http://127.0.0.1:8080'
CLASSIFIER_WEIGHT = 1
CLASSIFIER_BUFFER_LENGTH = 5
CLASSIFIER_IDENTIFIER = 'dev-classifier-00'
CLASSIFIER_POLL_INTERVAL_SEC = 1


async def task_collector(queue: asyncio.Queue, controller_url: str):
    try:
        while True:
            async with aiohttp.ClientSession() as session:
                params = dict(
                    id = CLASSIFIER_IDENTIFIER,
                    length = queue.qsize(),
                    maxlength = CLASSIFIER_BUFFER_LENGTH,
                    weight = CLASSIFIER_WEIGHT
                ).items()

                async with session.get(controller_url + '/allocate', params=list(params)) as resp:
                    if resp.ok:
                        rawcontent = await resp.text()
                        content = json.loads(rawcontent)
                        print(rawcontent)

                        if content["amount"] > 0:
                            for document in content["documents"]:
                                await queue.put(document)

            await asyncio.sleep(CLASSIFIER_POLL_INTERVAL_SEC)

    except asyncio.CancelledError:
        pass


async def main():
    background_tasks = set()

    controller_url = CLASSIFIER_CONTROLLER_URL
    mailbuffer = asyncio.Queue()

    # Create analizers
    spf_dkim_dmark_analyzer = Analyzer(set([FH12, FH13]))
    other_headers_alayzer = Analyzer(set([
        FH01, FH02, FH03, FH04, FH05, FH06, FH07, FH08, FH09,
        FH10, FH11, FH14, FH15, FH16, FH17, FH18, FH19, FH20,
        FH21, FH22, FH23, FH24
    ]))
    reputation_analyzer = Analyzer(set([FR01, FR02, FR03]))
    ml_analyzer = Analyzer(set([FML01, FML02]))

    # Create main classifier
    classifier = Classifier(
        [
            spf_dkim_dmark_analyzer,
            other_headers_alayzer,
            reputation_analyzer,
            ml_analyzer
        ],
        [ CR01, CR02, CR03, CR04, CR05, CR06 ]
    )

    # Run task gatherer and classifier
    collector = asyncio.create_task(task_collector(mailbuffer, controller_url))
    class_task = asyncio.create_task(classifier.run(mailbuffer, CLASSIFIER_IDENTIFIER, CLASSIFIER_CONTROLLER_URL))
    background_tasks.add(collector)
    background_tasks.add(class_task)

    await asyncio.gather(
        collector,
        class_task
    )

if __name__ == "__main__":
    asyncio.run(main())
