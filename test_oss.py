import oss2
from itertools import islice

auth = oss2.Auth('LTAIywfWTwuMjNno', 'orDMv6nloyf1u2pVGt2MufZM8FLjX3')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou-zmf.aliyuncs.com', 'cwl-oss-bucket')
for b in islice(oss2.ObjectIterator(bucket), 10):
    print(b.key)

bucket.get_object_to_file('tagging/model/tag2label.pkl', 'test.pkl')
