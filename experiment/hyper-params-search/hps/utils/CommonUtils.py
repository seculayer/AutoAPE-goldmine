# -*- coding: utf-8 -*-
# Author : Jin Kim
# e-mail : jinkim@seculayer.com
# Powered by Seculayer © 2020 Solution Development 2 Team, R&D Center. 

import os

# class : CommonUtils
class CommonUtils(object):
    @staticmethod
    def mkdir(dir_name):
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
