import utm
import rospy
from geometry_msgs.msg import Pose, QuaternionStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32

class PoseEstimator:
    def __init__(self):
        self.r = None
        self.t = None
        self.start_alt = None
        self.pub = rospy.Publisher('/current_pose', Pose, queue_size=10)
        self.gps_sub = rospy.Subscriber('/ublox/fix', NavSatFix, self.gps_cb)
        self.att_sub = rospy.Subscriber('/dji_osdk_ros/attitude', QuaternionStamped, self.att_cb)
        self.alt_sub = rospy.Subscriber('/dji_osdk_ros/height_above_takeoff', Float32, self.alt_sub)

    def gps_cb(self, msg):
        u = utm.from_latlon(msg.latitude, msg.longitude)

        pose = {'x': u[0],
                'y': u[1],
                'z': None}
        self.t = pose


    def att_cb(self, msg):
        # print("attitude callback")
        self.r = {'x': msg.quaternion.x,
                  'y': msg.quaternion.y,
                  'z': msg.quaternion.z,
                  'w': msg.quaternion.w}


    def rc_cb(self, msg):
        pass

    
    def get_pose(self):
        if self.r == None or self.t == None or self.t['x'] == None or self.t['y'] == None or self.t['z'] == None:
            return
        p = Pose()
        p.orientation.x = self.r['x']
        p.orientation.y = self.r['y']
        p.orientation.z = self.r['z']
        p.orientation.w = self.r['w']

        p.position.x = self.t['x']
        p.position.y = self.t['y']
        p.position.z = self.t['z']

        # print(p)

        self.pub.publish(p)

    def alt_sub(self, msg):
        if self.t == None:
            pose = {'x': None,
                    'y': None,
                    'z': msg.data}
            self.t = pose
        else:
            self.t['z'] = msg.data



rospy.init_node('py_odom')
r = rospy.Rate(4)
PE = PoseEstimator()
while not rospy.is_shutdown():
    # print('looping')
    PE.get_pose()
    r.sleep


