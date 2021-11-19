"""
This is a simple client for evaluating Punctaution and Capitalization models.
It uses riva API, installed with riva_api-1.6.0b0-py3-none-any.whl

Author: Marko B
Date: 19. 11. 2021
"""
import grpc
import argparse
import os
import riva_api.riva_nlp_pb2 as rnlp
import riva_api.riva_nlp_pb2_grpc as rnlp_srv

class BertPunctuatorClient(object):
    def __init__(self, grpc_server, model_name="riva_punctuation"):
        # generate the correct model based on precision and whether or not ensemble is used
        print("Using model: {}".format(model_name))
        self.model_name = model_name
        self.channel = grpc.insecure_channel(grpc_server)
        self.riva_nlp = rnlp_srv.RivaLanguageUnderstandingStub(self.channel)

        self.has_bos = True
        self.has_eos = False

    def run(self, input_strings):
        if isinstance(input_strings, str):
            # user probably passed a single string instead of a list/iterable
            input_strings = [input_strings]

        request = rnlp.TextTransformRequest()
        request.model.model_name = self.model_name
        for q in input_strings:
            request.text.append(q)
        response = self.riva_nlp.TransformText(request)

        return response.text[0]

def run_punct_capit(server,model,query):
    print("Client app to test punctuation and capitalization on Riva")
    client = BertPunctuatorClient(server, model_name=model)
    result = client.run(query)
    print(result)
    

run_punct_capit(server="34.90.177.137:50051",
                model="riva_punctuation",
                query="danes je lep dan v celju in ljubljani drugje pa ne jutri bo povsod slabo")
