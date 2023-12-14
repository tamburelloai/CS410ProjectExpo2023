from newsapi import NewsApiClient

class NewsAPIManager:
    def __init__(self, secret_key):
        self.key = secret_key
        self.client = NewsApiClient(api_key=secret_key)

    def __call__(self, *args, **kwargs):
        response = {'status': '', 'totalResults': 0, 'articles': []}
        query = kwargs.get('query')
        source_list = kwargs.get('sources')
        query_in_title_response = self.client.get_everything(qintitle=query,
                                          sources=source_list,
                                          language='en',
                                          sort_by='relevancy')

        query_in_body_response = self.client.get_everything(query,
                                                             sources=source_list,
                                                             language='en',
                                                             sort_by='relevancy')

        response['totalResults'] = query_in_title_response['totalResults'] + query_in_body_response['totalResults']
        response['articles'].extend(query_in_title_response['articles'])
        response['articles'].extend(query_in_body_response['articles'])
        return response






def test():
    import json
    newsapi_manager = NewsAPIManager('a90668840d4f498192aa7f9384e1fc7c')
    res = newsapi_manager("donald trump and joe biden")
    with open('newsapi_geteverything_sample.json', 'w') as json_file:
        json.dump(res, json_file)
    newsapi_manager.get_sources()

#test()

