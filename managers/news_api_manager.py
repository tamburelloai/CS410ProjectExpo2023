from newsapi import NewsApiClient

class NewsAPIManager:
    def __init__(self, secret_key):
        self.key = secret_key
        self.client = NewsApiClient(api_key=secret_key)

    def __call__(self, *args, **kwargs):
        query = kwargs.get('query')
        source_list = kwargs.get('sources')
        return self.client.get_everything(qintitle=query,
                                          sources=source_list,
                                          language='en',
                                          sort_by='relevancy')




def test():
    import json
    newsapi_manager = NewsAPIManager('a90668840d4f498192aa7f9384e1fc7c')
    res = newsapi_manager("donald trump and joe biden")
    with open('newsapi_geteverything_sample.json', 'w') as json_file:
        json.dump(res, json_file)
    newsapi_manager.get_sources()

#test()

