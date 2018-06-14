import MySQLdb
import json


def load_data(wrt_json_file):

    conn = MySQLdb.connect(host='192.168.1.12', user='root', passwd='reload123', db='shampoo')
    cur = conn.cursor()

    cur.execute('''select o.id, o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, i.url, i.id, o.type, i.src_src from object o inner join image i on o.img_id = i.id where i.verify_status = 1 and o.box_status = 0 \
        and i.id in (select distinct img_id from object where id in (select object_id from object_sames where status = 1) \
        or id in (select object_id from object_sames_9 where status = 1 ) \
        or id in (select object_id from object_sames_10 where status = 1) \
        or id in (select object_id from object_sames_11 where status = 1) \
        or id in (select same_object_id from object_sames where status = 1) \
        or id in (select same_object_id from object_sames_9 where status = 1) \
        or id in (select same_object_id from object_sames_10 where status = 1) \
        or id in (select same_object_id from object_sames_11 where status = 1))''')

    raw_data = [[int(i[0]), int(i[1]), int(i[2]), int(i[3]), int(i[4]), i[5], i[6], int(i[7]), i[8]] for i in cur.fetchall()]
    objectid_to_metadata = {str(o[0]): {'height':int(o[4]), 'width':int(o[3]), 'xmin':int(o[1]), 'ymin':int(o[2]), 'object_id': o[0], 'img_path': o[5], 'img_id': o[6]} for o in raw_data}

    with open(wrt_json_file, 'w') as f:
        f.write(json.dumps(objectid_to_metadata)) 
    
if __name__ == '__main__':
    load_data('../data/objectid_to_metadata_undist.json')
