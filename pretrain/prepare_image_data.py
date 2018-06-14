import MySQLdb
import json


def load_data(wrt_json_file):

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor() 

    cur.execute('''select distinct(i.url), i.id from object o inner join image i on o.img_id = i.id where i.verify_status = 1 and o.box_status = 0 \
        and i.id in (select distinct img_id from object where id in (select object_id from object_sames_orig_merge_test6_0402 where status = 1) \
        or id in (select object_id from object_sames_orig_merge_test7_0403 where status = 1 ) \
        or id in (select same_object_id from object_sames_orig_merge_test6_0402 where status = 1) \
        or id in (select same_object_id from object_sames_orig_merge_test7_0403 where status = 1))''')

    raw_data = [[i[0], i[1]] for i in cur.fetchall()]

    objectid_to_metadata = {o[1]: o[0] for o in raw_data}

    with open(wrt_json_file, 'w') as f:
        f.write(json.dumps(objectid_to_metadata)) 
    
if __name__ == '__main__':
    load_data('../data/image_data_mryx.json')
