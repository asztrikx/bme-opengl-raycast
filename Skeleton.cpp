//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Vörös Asztrik
// Neptun : WYZJ90
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

template<class T> struct Dnum {
	float f;
	T d;
	Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
	Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
	Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
	Dnum operator*(Dnum r) {
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
	return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}

typedef Dnum<vec2> Dnum2;

float floatEqual(float subject, float number) {
	float eps = 0.00001;
	return subject > number - eps && subject < number + eps;
}

vec3 perpendicular(vec3 v) {
	float coos[] = {v.x, v.y, v.z};
	float result[3];

	int zeroCount = 0;
	float sum = 0.0f;
	int lastNonZeroIndex = 0;
	for (int i = 0; i < 3; i++) {
		if (coos[i] == 0) {
			zeroCount++;
		} else {
			sum += coos[i];
			lastNonZeroIndex = i;
		}
		result[i] = 1;
	}

	sum -= coos[lastNonZeroIndex];
	result[lastNonZeroIndex] = -sum / coos[lastNonZeroIndex];
	return vec3(result[0], result[1], result[2]);
}

mat4 Identity() {
	return mat4(
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	);
}

const int tessellationLevel = 20;

struct Camera {
	vec3 wEye, wLookat, wVup;
	float fov, asp, fp, bp;
  public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 1; bp = 20;
	} 
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			                                       u.y, v.y, w.y, 0,
			                                       u.z, v.z, w.z, 0,
			                                       0,   0,   0,   1);
	}

	mat4 P() {
		return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
			        0,                      1 / tan(fov / 2), 0,                      0,
			        0,                      0,                -(fp + bp) / (bp - fp), -1,
			        0,                      0,                -2 * fp*bp / (bp - fp),  0);
	}
};

struct Material {
	vec3 kd, ks, ka;
	float shininess;
};

struct Light {
	vec3 La, Le; // TODO why La
	vec4 wPosition; // homogeneous coordinates, can be at ideal point
};

struct RenderState {
	mat4	           MVP, M, Minv, V, P;
	Material *         material;
	std::vector<Light> lights;
	vec3	           wEye;
};

struct Shader : public GPUProgram {
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wPosition, name + ".wPosition");
	}
};

class PhongShader : public Shader {
	const char * vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wPosition;
		};

		uniform mat4 MVP, M, Minv;
		uniform Light[8] lights;
		uniform int nLights;
		uniform vec3 wEye;

		layout(location = 0) in vec3  vtxPos;
		layout(location = 1) in vec3  vtxNorm;

		out vec3 wNormal;
		out vec3 wView;
		out vec3 wLight[8];

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP;
			
			vec4 wPosition = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wPosition.xyz * wPosition.w - wPosition.xyz * lights[i].wPosition.w;
			}
		    wView  = wEye * wPosition.w - wPosition.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	const char * fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wPosition;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;
		uniform int   nLights;

		in vec3 wNormal;
		in vec3 wView;
		in vec3 wLight[8];
		
        out vec4 fragmentColor;

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N; // TODO do it in vertex shader

			// TODO kd 2x volt texColorozva :D
			// TODO sky color was La in hf2
			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cosTheta = max(dot(N,L), 0), cosDelta = max(dot(N,H), 0);
				// TODO ambient is calculated for every light wtf
				radiance += material.ka * lights[i].La + 
                           (material.kd * cosTheta + material.ks * pow(cosDelta, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
  public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use();
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

/*Start of Geometry*/
class Geometry {
  protected:
	unsigned int vao, vbo;
  public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // This is unneeded by logic but probably used somewhere
	}

	virtual void Draw() = 0;

	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

class ParamSurface : public Geometry {
	// inline struct
	struct VertexData {
		vec3 position, normal;
	};

	unsigned int nVtxPerStrip, nStrips;
  public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++)
			glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
	}
};

struct Sphere : public ParamSurface {
	Sphere() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
	}
};

struct Paraboloid : public ParamSurface {
	vec3 eNormal = vec3(0,0,1);
	vec3 eP;
	vec3 start;
	float height;
	vec3 f;

	Paraboloid(float _height, float fDist) {
		height = _height;
		start = vec3(0,0,0);
		eP = start-fDist*eNormal;
		f = start+fDist*eNormal;

		create();
	}
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI;
		V = V * height;
		
		float dist = V.f - eP.z;
		float r = sqrtf(powf(dist,2) - powf(abs(V.f - f.z),2));
		X = Cos(U) * r;
		Y = Sin(U) * r;
		Z = V;
	}
};

struct Cylinder : public ParamSurface {
	Cylinder() { create(); }
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * 2 - 1.0f;
		X = Cos(U); Y = Sin(U); Z = V;
	}
};

/*Start of Object*/
// collection & transforms
struct Object {
	Shader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
	vec3 dir;
	
	Object(Shader * _shader, Material * _material, Geometry * _geometry)
	:scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0), dir(0,0,1) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		vec3 baseAxis(0,0,1);
		vec3 axis = cross(baseAxis, dir);
		float angle = acosf(dot(baseAxis, normalize(dir)));
		mat4 startRot, startRotInv;
		if (floatEqual(angle, 0.0f)) {
			startRot = startRotInv = Identity();
		} else {
			startRot = RotationMatrix(angle, axis);
			startRotInv = RotationMatrix(-angle, axis);
		}
		M = ScaleMatrix(scale) * startRot * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * startRotInv * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) {
		rotationAngle = 0.8f * tend; // TODO ez mi?
	}
};

class Scene {
	std::vector<Object *> objects;
	Camera camera;
	std::vector<Light> lights;
	
	//vec3 La = vec3(0.1f, 0.1f, 0.1f);
	//float epsilon = 0.005;

	vec3 viewUp = vec3(0,0,1);
	vec3 lookat = vec3(1,0,0);
	vec3 eye = vec3(7,0,5);

	float bigCylinderH = 0.1;
	float bigCylinderR = 0.5;
	float sphereR = 1.0f/8;
	float cylinderR = sphereR / 3;
	float paraH = 0.5, paraF = cylinderR + 0.1;
	float cylinderH0 = 2;
	float cylinderH1 = 1;
	vec3 dir0 = normalize(vec3(1,1,2));
	vec3 dir1 = normalize(vec3(-0.5,-1,2.8));
	vec3 paraDir = vec3(-2,-2,1);
	vec3 rot0 = normalize(vec3(1,1,1.5));
	vec3 rot1 = normalize(vec3(2,1,2));
	vec3 rot2 = normalize(vec3(2,2,1));
	vec3 joint0 = vec3(0,0,bigCylinderH);
	//vec3 sun = vec3(5,5,5);
	
	Object* cylinderObjStand;
	Object* cylinderObj0;
	Object* cylinderObj1;
	Object* sphereObj0;
	Object* sphereObj1;
	Object* sphereObj2;
	Object* paraboloidObj;

  public:
	void Recalc() {
		vec3 joint1 = joint0+cylinderH0*dir0;
		vec3 joint2 = joint1+cylinderH1*dir1;

		cylinderObj0->translation = joint0 + dir0*(cylinderH0/2);
		sphereObj1->translation = joint1;
		cylinderObj1->translation = joint1 + dir1*(cylinderH1/2);
		sphereObj2->translation = joint2;
		paraboloidObj->translation = joint2;

		paraboloidObj->rotationAxis = dir1;
		
		cylinderObj0->dir = dir0;
		cylinderObj1->dir = dir1;
		paraboloidObj->dir = paraDir;
	}

	void Build() {
		// Shaders
		Shader * phongShader = new PhongShader();

		// Materials
		Material * materialLamp = new Material;
		materialLamp->kd = vec3(55, 60, 63)/255.0f;
		materialLamp->ks = vec3(2,2,2);
		materialLamp->ka = vec3(0,0,0);
		materialLamp->shininess = 50;

		Material * materialPlane = new Material;
		materialPlane->kd = vec3(110, 76, 67)/255.0f;
		materialPlane->ks = vec3(0.1f,0.1f,0.1f);
		materialPlane->ka = vec3(0, 0, 0);
		materialPlane->shininess = 50;

		// Geometries
		Geometry* cylinder = new Cylinder();
		Geometry* sphere = new Sphere();
		Geometry* paraboloid = new Paraboloid(0.5, 0.14);

		cylinderObjStand = new Object(phongShader, materialLamp, cylinder);
		cylinderObj0 = new Object(phongShader, materialLamp, cylinder);
		cylinderObj1 = new Object(phongShader, materialLamp, cylinder);
		sphereObj0 = new Object(phongShader, materialLamp, sphere);
		sphereObj1 = new Object(phongShader, materialLamp, sphere);
		sphereObj2 = new Object(phongShader, materialLamp, sphere);
		paraboloidObj = new Object(phongShader, materialLamp, paraboloid);
		
		cylinderObjStand->scale = vec3(bigCylinderR,bigCylinderR,bigCylinderH/2);
		cylinderObj0->scale = vec3(cylinderR,cylinderR,cylinderH0/2);
		cylinderObj1->scale = vec3(cylinderR,cylinderR,cylinderH1/2);
		sphereObj0->scale = vec3(sphereR,sphereR,sphereR);
		sphereObj1->scale = sphereObj0->scale;
		sphereObj2->scale = sphereObj0->scale;

		cylinderObj0->rotationAxis = rot0;
		cylinderObj1->rotationAxis = rot1;

		cylinderObjStand->translation = vec3(0,0,bigCylinderH/2);
		sphereObj0->translation = joint0;

		objects.push_back(cylinderObjStand);
		objects.push_back(cylinderObj0);
		objects.push_back(cylinderObj1);
		objects.push_back(sphereObj0);
		objects.push_back(sphereObj1);
		objects.push_back(sphereObj2);
		objects.push_back(paraboloidObj);

		// Camera
		camera.wEye = eye;
		camera.wLookat = lookat;
		camera.wVup = viewUp;

		// Lights
		lights.resize(2);
		lights[0].wPosition = vec4(5, 5, 4, 0);
		lights[0].La = vec3(0.1f, 0.1f, 1);
		lights[0].Le = vec3(3, 0, 0);

		lights[1].wPosition = vec4(5, 10, 20, 0);
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(0, 3, 0);

		Recalc();
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		return;
		for (Object * obj : objects) obj->Animate(tstart, tend);
	}
};

Scene scene;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) { }

void onKeyboardUp(unsigned char key, int pX, int pY) { }

void onMouse(int button, int state, int pX, int pY) { }

void onMouseMotion(int pX, int pY) { }

void onIdle() {
	static float tend = 0;
	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
	}
	glutPostRedisplay();
}
