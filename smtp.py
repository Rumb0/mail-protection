
from datetime import datetime, timezone

from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase
)

from parsers import parse_mail_content


MAX_PENDING_MAILS = 20
MONGODB_HOST = '127.0.0.1'
MONGODB_PORT = 27017


class SMTPHandler:
    def getdatabase(self) -> AsyncIOMotorDatabase:
        try:
            return(self.database)
        except:
            self.mongoclient = AsyncIOMotorClient(MONGODB_HOST, MONGODB_PORT)
            self.database = self.mongoclient.mailfilter
            return(self.database)


    async def bypass_is_enabled(self, max_pending_mails=MAX_PENDING_MAILS) -> bool:
        database = self.getdatabase()
        qlength = await database.inbox_queue.estimated_document_count(maxTimeMS=200)
        return(qlength > max_pending_mails)


    async def handle_DATA(self, server, session, envelope) -> str:
        database = self.getdatabase()
        rawcontent = envelope.content.decode('utf8', errors='replace')
        content = parse_mail_content(rawcontent)

        if not await self.bypass_is_enabled():
            target_collection = database.inbox_queue
        else:
            target_collection = database.delivery_queue

        email = {
            "content": {
                "type": content["type"],
                "text": content["text"],
            },
            "headers": content["headers"],
            "sender": envelope.mail_from,
            "sender_opts": envelope.rcpt_options,
            "receivers": envelope.rcpt_tos,
            "options": envelope.mail_options
        }

        document = {
            "data": email,
            "source": { "type": "SMTP" },
            "timestamp": datetime.now(tz=timezone.utc)
        }

        result = await target_collection.insert_one(document)

        # print('Message from %s' % envelope.mail_from)
        # print('Message for %s' % envelope.rcpt_tos)
        # print('Message data:\n')
        # for ln in envelope.content.decode('utf8', errors='replace').splitlines():
        #     print(f'> {ln}'.strip())
        # print()
        # print('End of message')
        return '250 Message accepted for delivery'

