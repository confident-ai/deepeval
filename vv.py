from deepeval.confident.api import Api, HttpMethods, Endpoints

api = Api()
result = api.send_request(
    method=HttpMethods.GET,
    endpoint=Endpoints.TEST,
)

print(result)
