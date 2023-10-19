import json

DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты дружелюбный чат-бот для поиска продуктов, готовой еды и рецептов. Твоя задача - предоставить краткую и информативную помощь (не более 150 слов) на вопрос пользователя интернет-магазина продуктов здорового питания ВкуссВилл. Ты опираешься на поисковую выдачу и свои знания, всегда придерживаясь дружелюбного стиля. Главная цель - сделать поиск продуктов и кулинарных решений максимально полезным и соответствующим запросам пользователя, учитывая также вкусовые привычки, такие как веганство и вегетарианство. В случае, если поиск не дал результатов, ты всегда готов предложить альтернативные варианты или дать совет по выбору продуктов и рецептов. Не повторяй текст. Ответ должен быть непротиворечивым. \nПРИМЕР:\nQUESTION: бот найди чистящее средство\n=========\nSEARCH RESULT:\n- Средство чистящее для плит. Биоразлагаемое гипоаллергенное средство на основе природных компонентов. Подходит для очистки кухонных поверхностей, плит, духовок, сковородок и бытовых приборов. Средство экономично расходуется, устраняет неприятных запахи и удаляет жир даже в холодной воде. Его можно использовать для уборки в детских учреждениях. [^1^]\n- Универсальное средство Sanita. Предназначено для очистки кухонных поверхностей и устойчиво к действию щелочей. Подходит для плит, микроволновых печей и других поверхностей. Наносите средство, оставляйте на несколько секунд, затем протирайте влажной салфеткой и смойте водой. [^2^]\n- Крем чистящий Cif Актив Фреш. подходит для борьбы с известковым и мыльным налетом в ванной комнате. Этот крем содержит натуральные компоненты и чистящие микрокристаллы. Он проникает вглубь загрязнений, устраняет их и придает свежий аромат. Подходит для ухода за керамической плиткой и другими поверхностями. [^3^]\n=========\nFINAL ANSWER:\n1. Биоразлагаемое гипоаллергенное средство для плит и кухонных поверхностей, которое удаляет жир даже в холодной воде [^1^].\n2. Универсальное средство Sanita для очистки кухонных поверхностей, плит и микроволновых печей [^2^].\n3. Крем Cif Актив Фреш подходит для устранения известкового и мыльного налета в ванной, содержит натуральные компоненты [^3^].\nВыбор определяется типом поверхности и видом загрязнений."


class Conversation:
    def __init__(
            self,
            system_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
            user_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
            bot_message_template: str = DEFAULT_MESSAGE_TEMPLATE,
            system_prompt: str = DEFAULT_SYSTEM_PROMPT,
            system_role: str = "system",
            user_role: str = "user",
            bot_role: str = "bot",
            suffix: str = "<s>bot"
    ):
        self.system_message_template = system_message_template
        self.user_message_template = user_message_template
        self.bot_message_template = bot_message_template
        self.system_role = system_role
        self.user_role = user_role
        self.bot_role = bot_role
        self.suffix = suffix
        self.messages = [{
            "role": self.system_role,
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": self.user_role,
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": self.bot_role,
            "content": message
        })

    def count_tokens(self, tokenizer, current_messages):
        final_text = ""
        for message in current_messages:
            final_text += self.format_message(message)
        tokens = tokenizer([final_text])["input_ids"][0]
        return len(tokens)

    def shrink(self, tokenizer, messages, max_tokens):
        system_message = messages[0]
        other_messages = messages[1:]
        while self.count_tokens(tokenizer, [system_message] + other_messages) > max_tokens:
            other_messages = other_messages[2:]
        return [system_message] + other_messages

    def format_message(self, message):
        if message["role"] == self.system_role:
            return self.system_message_template.format(**message)
        if message["role"] == self.user_role:
            return self.user_message_template.format(**message)
        return self.bot_message_template.format(**message)

    def get_prompt(self, tokenizer, max_tokens: int = None, add_suffix: bool = True):
        messages = self.messages
        if max_tokens is not None:
            messages = self.shrink(tokenizer, messages, max_tokens)

        final_text = ""
        for message in messages:
            final_text += self.format_message(message)

        if add_suffix:
            final_text += self.suffix

        return final_text.strip()

    def iter_messages(self):
        for message in self.messages:
            yield self.format_message(message), message["role"]

    @classmethod
    def from_template(cls, file_name):
        with open(file_name, encoding="utf-8") as r:
            template = json.load(r)
        return Conversation(
            **template
        )

    def expand(self, messages, role_mapping=None):
        if not role_mapping:
            role_mapping = dict()

        if messages[0]["role"] == "system":
            self.messages = []

        for message in messages:
            self.messages.append({
                "role": role_mapping.get(message["role"], message["role"]),
                "content": message["content"]
            })
