from gdeltdoc import GdeltDoc, Filters


class GDELTManager:
    def __init__(self, secret):
        self.secret = secret
        self.client = GdeltDoc()

    def __call__(self, *args, **kwargs):
        pass


def test():
    gdelt_manager = GDELTManager(secret='null')
    f = Filters(
        keyword = "climate change",
        start_date = "2020-05-10",
        end_date = "2020-05-11"
    )
    articles = gdelt_manager.client.article_search(f)

    # Get a timeline of the number of articles matching the filters
    timeline = gdelt_manager.client.timeline_search("timelinevol", f)

    print(articles)
    print(timeline)

