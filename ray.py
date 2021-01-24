import numpy as np

from utils import *


"""Core implementation of the ray tracer, containing classes with which to represent various
geometries (Sphere, Mesh, etc.) that define the contents of scenes, as well as classes (Ray,
Hit) and functions (shade) used to render them, which may be done by calling render_image.
"""

class Ray:

	def __init__(self, origin, direction, start=0., end=np.inf):
		"""Create a ray with the given origin and direction.
		Parameters:
		  origin : (3,) - 3D point representing the start point of the ray
		  direction : (3,) - 3D vector representing the direction of the ray, may not be normalized
		  start, end : float - the minimum and maximum t values to check for intersections (prevents self-clipping)
		"""
		# converted to double for precision
		self.origin = np.array(origin, np.float64)
		self.direction = np.array(direction, np.float64)
		self.start = start
		self.end = end


class Material:

	def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
		"""Create a new material with the given parameters.
		Parameters:
		  k_d : float, (3,) or (h,w,3) - the diffuse coefficient
		  k_s : float, (3,) or (h,w,3) - the specular coefficient
		  p : float or (h,w)-- the specular exponent
		  k_m : float, (3,) or (h,w,3) - the mirror reflection coefficient
		  k_a : float, (3,) or (h,w,3) - the ambient coefficient (defaults to match diffuse color)
		"""
		self.k_d = k_d
		self.k_s = k_s
		self.p = p
		self.k_m = k_m
		self.k_a = k_a if k_a is not None else k_d

	def update(self,u,v):
		"""Calculates material properties, taking textures into account. This does not change the Material object.
		Parameters:
		  u,v - texture coordinates
		Returns:
		  (5,) - material properties recalculated with textures
		"""
		k_d = self.k_d
		k_s = self.k_s
		p = self.p
		k_m = self.k_m
		k_a = self.k_a if self.k_a is not None else self.k_d
		if isinstance(k_d,np.ndarray) and k_d.shape!=(3,):
			dims = k_d.shape
			i = np.remainder(int(dims[1]*u),dims[1])
			j = np.remainder(int(dims[0]*v),dims[0])
			k_d = k_d[j][i]
		if isinstance(k_s,np.ndarray) and k_s.shape!=(3,):
			dims = k_s.shape
			i = np.remainder(int(dims[1]*u),dims[1])
			j = np.remainder(int(dims[0]*v),dims[0])
			k_s = k_s[j][i]
		if isinstance(p,np.ndarray):
			dims = p.shape
			i = np.remainder(int(dims[1]*u),dims[1])
			j = np.remainder(int(dims[0]*v),dims[0])
			p = p[j][i]
		if isinstance(k_m,np.ndarray) and k_m.shape!=(3,):
			dims = k_m.shape
			i = np.remainder(int(dims[1]*u),dims[1])
			j = np.remainder(int(dims[0]*v),dims[0])
			k_m = k_m[j][i]
		if isinstance(k_a,np.ndarray) and k_a.shape!=(3,):
			dims = k_a.shape
			i = np.remainder(int(dims[1]*u),dims[1])
			j = np.remainder(int(dims[0]*v),dims[0])
			k_a = k_a[j][i]
		return [k_d,k_s,p,k_m,k_a]


class Hit:

	def __init__(self, t, point=None, normal=None, uv=None, material=None):
		"""Create a Hit with the given parameters.
		Parameters:
		  t : float - the t value of the intersection along the ray
		  point : (3,) - the point where the intersection happens
		  normal : (3,) - the outward-facing unit normal to the surface at the hit point
		  uv : (2,) - the texture coordinates at the intersection point
		  material : (Material) - the material of the surface
		"""
		self.t = t
		self.point = point
		self.normal = normal
		self.uv = uv
		self.material = material

# value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

	def __init__(self, center, radius, material):
		"""Create a sphere with the given center, radius, and material.
		Parameters:
		  center : (3,) - a 3D tuple representing the sphere's center
		  radius : float - a float specifying the sphere's radius
		  material : Material - the material of the surface
		"""
		self.center = center
		self.radius = radius
		self.material = material

	def intersect(self, ray):
		"""Computes the first intersection between a ray and this sphere.
		Parameters:
		  ray : Ray - the ray to intersect with the sphere
		Returns:
		  Hit - the hit between the ray and the surface of the sphere
		"""
		e, d, c, r = ray.origin, ray.direction, self.center, self.radius
		D = (d @ (e - c))**2 - (d @ d) * ((e - c) @ (e - c) - r**2)
		if D < 0:
			return no_hit
		t = (-d @ (e - c) - np.sqrt(D)) / (d @ d)
		if t < ray.start:
			t = (-d @ (e - c) + np.sqrt(D)) / (d @ d)
		if ray.start < t < ray.end:
			point = ray.origin + t * ray.direction
			normal = normalize(self.center - point)
			u = 0.5 + np.arctan2(-normal[0], -normal[2]) / (2 * np.pi)
			v = 0.5 - np.arcsin(normal[1]) / np.pi
			return Hit(t, point, -normal, vec([u,v]), self.material)
		else:
			return no_hit


class Triangle:

	def __init__(self, vs, material):
		"""Create a triangle from the given vertices.
		Parameters:
		  vs : (3,3) - an array of 3 3D points representing the vertices, ordered counterclockwise as seen from the outside
		  material : Material - the material of the surface
		"""
		self.vs = vs
		self.material = material
		self.norm_vec = normalize(np.cross(vs[1] - vs[0], vs[2] - vs[0]))

	def intersect(self, ray):
		"""Computes the intersection between a ray and this triangle, if it exists.
		Parameters:
		  ray : Ray - the ray to intersect with the triangle
		Returns:
		  Hit - the hit where the ray is incident with the surface of the triangle
		"""
		vs = self.vs
		A = np.array([vs[0] - vs[1], vs[0] - vs[2], ray.direction]).transpose()
		b = vs[0] - ray.origin
		beta, gamma, t = np.linalg.solve(A, b)
		if beta > 0 and gamma > 0 and beta + gamma < 1 and ray.start < t < ray.end:
			return Hit(t, ray.origin + t * ray.direction, self.norm_vec, vec([beta,gamma]), self.material)
		else:
			return no_hit


class Mesh:

	def __init__(self, inds, posns, normals, uvs, material):
		"""Create a mesh to describe complex multi-triangle surfaces or solids
		For a given triangle i, posns[inds[i][0:2]] are the vertex locations in CCW 
		order, normals[i] is the corresponding normal vector (which may not be unit),
		and uvs[inds[i][0:2]] are the texture coordinates of the vertices in CCW order.

		Parameters:
			inds - list of (3,) integer tuples of point indexes describing each triangle in counterclockwise order
			posns - list of (3,) 3D coordinates describing the location of each vertex
			normals - list of (3,) 3D outward facing normal vectors for each surface
			uvs - list of (2,) 2D texture coordinates for each vertex
			material : (Material) - material of the surface
		"""
		self.inds = np.array(inds, np.int32)
		self.posns = np.array(posns, np.float32)
		self.normals = np.array(normals, np.float32) if normals is not None else None
		self.uvs = np.array(uvs, np.float32) if uvs is not None else None
		self.material = material

	def intersect(self, ray):
		"""Computes the intersection between a ray and this mesh, if it exists.
			Use the batch_intersect function in the utils package
		Parameters:
			ray : Ray - the ray to intersect with the mesh
		Returns:
			Hit - the incident hit
		"""
		triangles = [] 
		for (a,b,c) in self.inds:
			triangles.append([self.posns[a], self.posns[b], self.posns[c]])
		(t, beta, gamma, i) = batch_intersect(np.asarray(triangles), ray)
		if i == -1:
			return no_hit
		alpha = 1. - beta - gamma 
		hit_loc = ray.origin + t * ray.direction
		triangle = triangles[i]
		indices = self.inds[i]
		# print(indices)
		if self.normals is None:
			normal = normalize(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[1]))
		else:
			normal = normalize(alpha*self.normals[indices[0]] + beta*self.normals[indices[1]] + gamma*self.normals[indices[2]])
		if self.uvs is None:
			uv = np.asarray((beta, gamma))
		else:
			uv = alpha*self.uvs[indices[0]] + beta*self.uvs[indices[1]] + gamma*self.uvs[indices[2]]
		return Hit(t, hit_loc, normal, uv, self.material)


class Camera:

	def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]),
				 vfov=90.0, aspect=1.0):
		"""Create a camera with given viewing parameters.
		Parameters:
		  eye : (3,) - 3D point representing the camera's location
		  target : (3,) - where the camera is looking as a 3D point that appears centered in the view
		  up : (3,) - the camera's orientation as a 3D vector pointing straight up in the view
		  vfov : float - the full vertical field of view in degrees
		  aspect : float - the aspect ratio of the camera's view (ratio of width to height)
		"""
		self.eye = eye
		self.f = 1 / np.tan(vfov/2 * np.pi/180)
		self.aspect = aspect
		w = normalize(eye - target)
		u = normalize(np.cross(up, w))
		v = np.cross(w, u)
		self.M = np.block([[u, 0], [v, 0], [w, 0], [eye, 1]]).transpose()

	def generate_ray(self, img_point):
		"""Compute the ray corresponding to a point in the image.
		Parameters:
		  img_point : (2,) - 2D point in the unit square where (0,0) is the lower left corner
		Returns:
		  Ray - The ray corresponding to that image location (not necessarily normalized)
		"""
		dir = np.array([self.aspect * (2*img_point[0] - 1), 2*img_point[1] - 1, -self.f, 0])
		return Ray(self.eye, (self.M @ dir)[:3])


class PointLight:

	def __init__(self, position, intensity):
		"""Create a point light at given position and with given intensity
		Parameters:
		  position : (3,) - 3D point giving the light source location in scene
		  intensity : (3,) or float - RGB or scalar intensity of the light source
		"""
		self.position = position
		self.intensity = intensity

	def illuminate(self, ray, hit, scene):
		"""Compute the shading at a surface point due to this light.
		Parameters:
		  ray : Ray - the ray that hit the surface
		  hit : Hit - the hit data
		  scene : Scene - the scene, used for shadow rays
		Returns:
		  (3,) - the light reflected from the surface
		"""
		offset = self.position - hit.point
		shad_hit = scene.intersect(Ray(hit.point, offset, 1e-6, 1.))
		if shad_hit.t > 1.0:
			mat = hit.material
			r = np.linalg.norm(offset)
			L = offset / r
			N = normalize(hit.normal)
			irrad = self.intensity * np.maximum(0, N @ L) / (r**2)
			V = -normalize(ray.direction)
			H = normalize(V + L)
			u = hit.uv[0]
			v = hit.uv[1]
			tvals = mat.update(u,v)
			return irrad * (tvals[0] + tvals[1] * np.maximum(0, H @ N) ** tvals[2])
		else:
			return vec([0,0,0])


class AmbientLight:

	def __init__(self, intensity):
		"""Create an ambient light of given intensity
		Parameters:
		  intensity (3,) or float: the intensity for the ambient light
		"""
		self.intensity = intensity

	def illuminate(self, ray, hit, scene):
		"""Compute the shading at a surface point due to this light.
		Parameters:
		  ray : Ray - the ray that hit the surface
		  hit : Hit - the hit data
		  scene : Scene - the scene, used for shadow rays
		Returns:
		  (3,) - the light reflected from the surface
		"""
		mat = hit.material
		u = hit.uv[0]
		v = hit.uv[1]
		tvals = mat.update(u,v)
		return tvals[4] * self.intensity


class Scene:

	def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
		"""Create a scene containing the given objects.

		Parameters:
		  surfs : [Sphere, Triangle] - list of the surfaces in the scene
		  bg_color : (3,) - RGB color that is seen where no objects appear
		"""
		self.surfs = surfs
		self.bg_color = bg_color

	def intersect(self, ray):
		"""Computes the first (smallest t) intersection between a ray and the scene.

		Parameters:
		  ray : Ray - the ray to intersect with the scene
		Returns:
		  Hit - the hit data
		"""
		hits = (surf.intersect(ray) for surf in self.surfs)
		return min((hit for hit in hits), key=lambda h: h.t, default=no_hit)


def reflect(v, n):
	return 2 * (v @ n) * n - v

MAX_DEPTH = 4

def shade(ray, hit, scene, lights, depth=0):
	"""Compute shading for a ray-surface intersection.
	Parameters:
	  ray : Ray - the ray that hit the surface
	  hit : Hit - the hit data
	  scene : Scene - the scene
	  lights : [PointLight or AmbientLight] - the lights
	  depth : int - the recursion depth so far
	Returns:
	  (3,) - the color seen along this ray
	When mirror reflection is being computed, recursion will only proceed to a depth
	of MAX_DEPTH, with nothing beyond that depth.
	"""
	mat = hit.material
	u = hit.uv[0]
	v = hit.uv[1]
	tvals = mat.update(u,v)
	direct = sum(light.illuminate(ray, hit, scene) for light in lights)
	reflected = 0
	if np.any(tvals[3]) > 0 and depth < MAX_DEPTH:
		refl_dir = reflect(-normalize(ray.direction), hit.normal)
		refl_ray = Ray(hit.point, refl_dir, 5e-5)
		refl_hit = scene.intersect(refl_ray)
		if refl_hit.t < np.inf:
			reflected = tvals[3] * shade(refl_ray, refl_hit, scene, lights, depth+1)
		else:
			reflected = tvals[3] * scene.bg_color
	return direct + reflected


def render_image(camera, scene, lights, nx, ny):
	"""Render a ray traced image.
	Parameters:
	  camera : Camera - the camera defining the view
	  scene : Scene - the scene to be rendered
	  lights : Lights - the lights illuminating the scene
	  nx, ny : int - the dimensions of the rendered image
	Returns:
	  (ny, nx, 3) float32 - the RGB image
	"""
	cam_img = np.zeros((ny,nx,3), np.float32)
	for i in range(ny):
		for j in range(nx):
			ray = camera.generate_ray(np.array([j/nx, i/ ny]))
			hit = scene.intersect(ray)
			cam_img[i,j] = shade(ray, hit, scene, lights) if hit.t < np.inf else scene.bg_color
	return cam_img
