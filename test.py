from flask import Flask
from flask import render_template,request
from pymongo import MongoClient
import json
from bson import json_util
from bson.json_util import dumps

app = Flask(__name__,template_folder='/home/sri/Downloads/AAL-94_dataset/AALtorch/template')
print(app)
MONGOD_HOST = 'localhost'
MONGOD_PORT = 27017
DBS_NAME = 'pymongo_test'
COLLECTION_NAME = 'posts'
FIELDS = {'Class': True, 'Conf': True,'Dates': True}
#var date,timestamp


@app.route("/")
def demo1():
    return render_template("demo.html")

@app.route("/pymongo_test/posts")
def donor_projects():
    connection = MongoClient(MONGOD_HOST, MONGOD_PORT)
    collection = connection[DBS_NAME][COLLECTION_NAME]
    projects = collection.find(projection=FIELDS)
    #print(timestamp)
    cursor = []
    cursor = collection.aggregate([{"$group":{"_id":"$Class","Count":{"$sum":1}}}])
    json_projects = []
    count = []
    for document in cursor:
    	count.append(document)
    for project in projects:
        json_projects.append(project)
    json_projects = json.dumps(json_projects, default=json_util.default)
    count = json.dumps(count, default=json_util.default)
    #print(count)
    connection.close()
    return json_projects



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)
    


