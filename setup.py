from distutils.core import setup
from Cython.Build import cythonize

setup(name="detect", ext_modules=cythonize(["mdetect.py",
                                            "mysocket/my_socket.py",
                                            "strategy/base_strategy.py", "strategy/strategy.py", "strategy/strategy.py", "strategy/todo.py",
                                            "camera/my_camera.py"]))

# setup(name="detect", ext_modules=cythonize(["mdetect.py"]))