from features.core import BaseFeature, FeatureType


class FH01(BaseFeature):
    DESCRIPTION = "Empty 'From' header"

    def check(self):
        if not self.envelope.senders:
            return True

        if len(self.envelope.senders) == 1 and not self.envelope.senders[0].email:
            return True

        return False


class FH02(BaseFeature):
    DESCRIPTION = "No Real Sender"

    def check(self):
        if not self.envelope.real_sender:
            return True

        if not self.envelope.real_sender.email:
            return True

        return False


class FH03(BaseFeature):
    DESCRIPTION = "External Email (by real sender)"
    BLOCKED_BY = [FH02]

    def check(self):
        if not self.envelope.real_sender.domain:
            return True
        return not is_owned_domain(self.envelope.real_sender.domain, self.domains)


class FH04(BaseFeature):
    DESCRIPTION = "Internal Email (by real sender)"
    BLOCKED_BY = [FH02]

    def check(self):
        if not self.envelope.real_sender.domain:
            return False
        return is_owned_domain(self.envelope.real_sender.domain, self.domains), self.envelope.real_sender.email


class FH05(BaseFeature):
    DESCRIPTION = "Real sender domain is different than 'From' domains"
    BLOCKED_BY = [FH02, FH01]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        sender_domain = self.envelope.real_sender.domain
        from_domains = [sender.domain for sender in self.envelope.senders if sender.domain]
        res = not is_owned_domain(sender_domain, from_domains)

        if res:
            return True
        else:
            return False


class FH06(BaseFeature):
    DESCRIPTION = "No 'Reply-to' address"

    def check(self):
        return not self.envelope.reply_to


class FH07(BaseFeature):
    DESCRIPTION = "External 'Reply-to'"
    BLOCKED_BY = [FH06]

    def check(self):
        return not is_owned_domain(self.envelope.reply_to.domain, self.domains)


class FH08(BaseFeature):
    DESCRIPTION = "Similar but not matching domains of real sender and one of protected organization domains"
    REQUIRES = [FH03]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        return is_similar_to_owned_domain(self.envelope.real_sender.domain, self.domains)


class FH09(BaseFeature):
    DESCRIPTION = "Internal email (by 'From' sender)"
    BLOCKED_BY = [FH01]

    def check(self):
        for sender in self.envelope.senders:
            if not sender.domain:
                continue
            if is_owned_domain(sender.domain, self.domains):
                return True
        return False


class FH10(BaseFeature):
    DESCRIPTION = "Similar but not matching domains of 'From' header and protected organization domains"
    BLOCKED_BY = [FH09]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        for sender in self.envelope.senders:
            if not sender.domain:
                continue
            return is_similar_to_owned_domain(sender.domain, self.domains)

        return False


class FH11(BaseFeature):
    DESCRIPTION = "No information about real sender's domain"
    BLOCKED_BY = [FH02]
    TYPE = FeatureType.SPAM

    def check(self):
        if self.envelope.real_sender.domain:
            res = get_domain_creation_date(self.envelope.real_sender.domain)

            if res["success"]:
                return True
        return False


class FH12(BaseFeature):
    DESCRIPTION = "SPF Check Failed"
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        return self.envelope.auth.spf.verdict is False


class FH13(BaseFeature):
    DESCRIPTION = "DKIM Check Failed"

    def check(self):
        return self.envelope.auth.dkim.verdict is False


class FH14(BaseFeature):
    DESCRIPTION = "Receiver company name in 'From', but email is external"
    BLOCKED_BY = [FH01]
    REQUIRES = [FH03]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        try:
            c_domain = self.envelope.real_receiver.domain
            d = tldextract.extract(c_domain).domain
            c_name = d.split('.')[-1]
        except Exception:
            return False

        if not self.envelope.senders[0].name:
            return False

        s_words = re.sub(r"[^\w]", " ", self.envelope.senders[0].name).split()
        s_words = list(map(lambda x: x.lower(), s_words))

        if c_name.lower() in s_words or c_domain in self.envelope.senders[0].name:
            return True

        return False


class FH15(BaseFeature):
    DESCRIPTION = "'Reply-to' domain is lookalike to sender`s domain"
    REQUIRES = [FH07]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        if not self.envelope.reply_to.domain:
            return False

        for sender in self.envelope.senders:
            if not sender.domain:
                continue
            if get_first_level_domain(sender.domain) == get_first_level_domain(self.envelope.reply_to.domain):
                return False
            if sender.domain.endswith('.' + self.envelope.reply_to.domain):
                return False
            if self.envelope.reply_to.domain.endswith('.' + sender.domain):
                return False
            if is_similar_to_owned_domain(self.envelope.reply_to.domain, [sender.domain]):
                return True

        return False


class FH16(BaseFeature):
    DESCRIPTION = "The sender's domain from header contains forbidden symbols"
    BLOCKED_BY = [FH01]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        for sender in self.envelope.senders:
            if not sender.domain:
                continue
            try:
                idna.decode(sender.domain)
            except idna.InvalidCodepoint:
                return True
        return False


class FH17(BaseFeature):
    DESCRIPTION = "'Reply-to' login is the same as the 'From' login"
    BLOCKED_BY = [FH01, FH06]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        senders_logins = [s.login for s in self.envelope.senders if s.login]
        rto_login = self.envelope.reply_to.login

        if not rto_login:
            return False

        if rto_login in senders_logins:
            return True

        return False


class FH18(BaseFeature):
    DESCRIPTION = "Real sender's name in email ends with the protected domain"
    BLOCKED_BY = [FH02]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        for d in self.domains:
            domain = tldextract.extract(d).domain

            if len(domain) < 3:
                continue

            if self.envelope.real_sender.email.split('@')[0].endswith('.' + domain):
                return True

        return False


class FH19(BaseFeature):
    DESCRIPTION = "Real sender's domain was registered quite recently"
    BLOCKED_BY = [FH02, FH11]
    TYPE = FeatureType.SPAM

    def check(self):
        if self.envelope.real_sender.domain:
            res = get_domain_creation_date(self.envelope.real_sender.domain)
            if not res["success"] or not res["data"]:
                return False

            creation_date = res["data"]

            if not isinstance(creation_date, datetime):
                creation_date = creation_date[-1]

            days_ago = (datetime.now() - creation_date).days
            if days_ago // 30 <= 1:  # months
                return True

        return False


class FH20(BaseFeature):
    DESCRIPTION = "The subject of the email is about payment transactions"
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        text = self.envelope.headers.subject
        if not text:
            return False

        text = text.lower()

        payment_dict = [
            re.compile(r'[Pp]ayment'),
            re.compile(r'[Оо]плат|[Пп]лат[её]ж'),
            re.compile(r'PAYMENT'),
            re.compile(r'ОПЛАТА|ПЛАТ[ЕЁ]Ж'),
            re.compile(r'[Tt]ransaction'),
            re.compile(r'[Тт]ран[зс]акция|[Сс]делка'),
            re.compile(r'TRANSACTION'),
            re.compile(r'ТРАН[ЗС]АКЦИЯ|СДЕЛКА'),
            re.compile(r'[Dd]ebt'),
            re.compile(r'[Дд]олг'),
            re.compile(r'DEBT'),
            re.compile(r'ДОЛГ'),
        ]

        for reg in payment_dict:
            if reg.search(text):
                return True

        return False


class FH21(BaseFeature):
    DESCRIPTION = "The subject of the email calls for urgent action"
    TYPE = FeatureType.SPAM

    def check(self):
        text = self.envelope.headers.subject
        if not text:
            return False

        text = text.lower()

        urgent_dict = [
            re.compile(r'[Uu]rgent'),
            re.compile(r'[Сс]рочн(о|ый)'),
            re.compile(r'URGENT'),
            re.compile(r'СРОЧН(О|ЫЙ)'),
            re.compile(r'[Ii]mmediate'),
            re.compile(r'[Нн]емедленн(о|ый)'),
            re.compile(r'IMMEDIATE'),
            re.compile(r'НЕМЕДЛЕНН(О|ЫЙ)'),
            re.compile(r' now to '),
            re.compile(r'[Oo]utstanding'),
        ]

        for reg in urgent_dict:
            if reg.search(text):
                return True

        return False


class FH22(BaseFeature):
    DESCRIPTION = "The real sender's domain has no suffix"
    BLOCKED_BY = [FH02]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        if self.envelope.real_sender.domain and '.' not in self.envelope.real_sender.domain:
            return True

        return False


class FH23(BaseFeature):
    DESCRIPTION = "The subject of the email contains impersonation"
    BLOCKED_BY = [FH04]
    TYPE = FeatureType.IMPERSONATION

    def check(self):
        if self.envelope.real_receiver and self.envelope.real_receiver.domain and \
                self.envelope.headers.subject:

            domain = tldextract.extract(self.envelope.real_receiver.domain).domain

            if f"from {domain}" in self.envelope.headers.subject.lower():
                return True

        return False


class FH24(BaseFeature):
    DESCRIPTION = "The subject say`s about the password expiration"

    def check(self):
        text = self.envelope.headers.subject

        if not text:
            return False

        if re.search(r'password [\w ]{0,15}expir', text.lower()):
            return True

        return False
