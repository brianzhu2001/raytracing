from utils import *
from ray import *
from cli import render

green = Material(vec([0.2, 0.8, 0.2]))

creeper = Material(load_image('textures/02_-_Default_baseColor.png'))

(i, p, n, t) = read_obj(open("models/creeper.obj"))

scene = Scene(
	[Sphere(vec([0,-147.75,0]), 141.5, green),] 
	+[Mesh(i, 0.5*p, None, t, creeper),])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([15,8.5,25]), target=vec([0,0,0]), vfov=30, aspect=16/9)

render(camera, scene, lights)
