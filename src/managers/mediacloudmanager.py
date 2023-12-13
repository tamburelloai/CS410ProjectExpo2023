import mediacloud.api

class MediaCloudManager:
    def __init__(self, secret):
        self.secret = secret
        self.client = mediacloud.api.DirectoryApi(self.secret)
        #self.client = mediacloud.api.BaseApi(self.secret)

    def __call__(self, *args, **kwargs):
        INDIA_NATIONAL_COLLECTION = 34412118
        SOURCES_PER_PAGE = 100  # the number of sources retrieved per page

        sources = []
        offset = 0   # offset for paging through
        while True:
            # grab a page of sources in the collection
            response = self.client.source_list(collection_id=INDIA_NATIONAL_COLLECTION, limit=SOURCES_PER_PAGE, offset=offset)
            # add it to our running list of all the sources in the collection
            sources += response['results']
            # if there is no next page then we're done so bail out
            if response['next'] is None:
                break
            # otherwise setup to fetch the next page of sources
            offset += len(response['results'])
        print("India National Collection has {} sources".format(len(sources)))