from vanna.flask import VannaFlaskApp

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': 'gpt-oss'})

app = VannaFlaskApp(vn)
app.run()