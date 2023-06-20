import json
import pymongo

from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from aiohttp import web


from classifier.classifier import ClassifierVerdict


DOCUMENT_BATCH_SIZE = 5
MONGODB_HOST = '127.0.0.1'
MONGODB_PORT = 27017


routes = web.RouteTableDef()
mongoclient = AsyncIOMotorClient(MONGODB_HOST, MONGODB_PORT)
database = mongoclient.mailfilter

inbox_queue = database.inbox_queue
analysis_queue = database.analysis_queue
deliver_queue = database.deliver_queue
workers = database.workers


@routes.get('/allocate')
async def allocate(request):
    classifier_identifier = request.query["id"]
    classifier_queue_length = int(request.query["length"])
    classifier_weight = int(request.query["weight"])

    query_mails = DOCUMENT_BATCH_SIZE - classifier_queue_length
    if query_mails <= 0:
        return web.Response(text=json.dumps(dict(amount=0, documents={})))

    documents = await inbox_queue.find(
        sort=[("timestamp", pymongo.DESCENDING)],
        limit=query_mails
    ).to_list(length=None)

    for doc in documents:
        doc["classifier"] = {}
        doc["classifier"]["id"] = classifier_identifier
        doc["classifier"]["weight"] = classifier_weight

        await inbox_queue.delete_one({ "_id": doc["_id"] })
        await analysis_queue.insert_one(doc)

        doc["data"]["id"] = str(doc["_id"])

    update_result = await workers.find_one_and_update(
        {
            "identifier": classifier_identifier
        },
        {
            "$set": {
                "identifier": classifier_identifier,
                "queue_length": classifier_queue_length,
                "weight": classifier_weight,
            },
            "$inc": {
                "querried_analisys": len(documents)
            }
        },
        projection={ "_id": False },
        upsert=True
    )
    print(update_result)

    pyload = dict(
        amount = len(documents),
        documents = [ d["data"] for d in documents ]
    )

    return web.Response(text=json.dumps(pyload))


@routes.post('/submit')
async def submit(request):
    data = await request.json()

    result = dict(
        verdict = data["verdict"],
        classifier = data["cid"]
    )

    doc = await analysis_queue.find_one_and_delete(
        { "_id": ObjectId(data["mid"]) }
    )
    doc["result"] = result

    if data["verdict"] != ClassifierVerdict.MALICIOUS:
        await deliver_queue.insert_one(doc)

    return web.Response()


@routes.get('/health')
async def health(request):
    statuses = await workers.find(projection={ "_id": False }).to_list(length=None)

    return web.Response(text=json.dumps(statuses, default=str))
