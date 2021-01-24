from utils import *
from ray import *
from cli import render

green = Material(vec([0.2, 0.8, 0.2]))

sheep = Material(load_image('textures/Sheep_baseColor.png'))
sheepfur = Material(load_image('textures/Sheep_Fur_baseColor.png'))

(i_sheep, p_sheep, n_sheep, t_sheep) = read_obj(open("models/sheep2.obj"))
(i_sheepfur, p_sheepfur, n_sheepfur, t_sheepfur) = read_obj(open("models/sheep_fur.obj"))

scene = Scene(
	[Sphere(vec([0,-147.75,0]), 141.5, green),] 
	+[Mesh(i_sheep, 0.5*p_sheep, None, t_sheep, sheep),] 
    +[Mesh(i_sheepfur, 0.5*p_sheepfur, None, t_sheepfur, sheepfur),])

lights = [
    PointLight(vec([12,10,5]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([15,8.5,25]), target=vec([0,0,0]), vfov=30, aspect=16/9)

render(camera, scene, lights)
