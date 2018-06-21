import os
import setuptools
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
import numpy

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
        
def locate_cuda():
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = os.path.join(home, "bin", "nvcc")
    else:
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError("The nvcc binary could not be located in your $PATH.")
        home = os.path.dirname(os.path.dirname(nvcc))

    # cudaconfig = {"home": home,
    #               "nvcc": nvcc,
    #               "include": os.path.join(home, "include"),
    #               "lib": os.path.join(home, "lib")}
    cudaconfig = {"home": home,
                  "nvcc": nvcc,
                  "include": os.path.join(home, "include")}
    if os.path.exists(os.path.join(cudaconfig["home"], "lib")):
        cudaconfig["lib"] = os.path.join(cudaconfig["home"], "lib")
    else:
        cudaconfig["lib"] = os.path.join(cudaconfig["home"], "lib64")
    print(cudaconfig)
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError("The CUDA {0} path could not be located in {1}".format(k, v))
    return cudaconfig

CUDA = locate_cuda()
print(CUDA)
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if find_in_path("swig", os.environ["PATH"]):
    subprocess.check_call("swig -python -c++ -o src/emc_cuda_swig_wrap.cpp src/emc_cuda.i", shell=True)
    subprocess.check_call("swig -python -c++ -o src/emc_cpu_swig_wrap.cpp src/emc_cpu.i", shell=True)
else:
    raise EnvironmentError("The swig executable was not found in your PATH")
    
ext_cuda = Extension("_emc_cuda",
                     sources=["src/emc_cuda_swig_wrap.cpp",
                              "src/emc_cuda.cu",
                              "src/calculate_responsabilities_cuda.cu",
                              "src/calculate_scaling_cuda.cu",
                              "src/update_slices_cuda.cu"],
                     library_dirs=[CUDA["lib"]],
                     libraries=["cudart"],
                     runtime_library_dirs=[CUDA["lib"]],
                     extra_compile_args={"clang": [],
                                         "nvcc": ["--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"]},
                     #"nvcc": ["-arch=sm_20", "--ptxas-options=-v", "-c", "--compiler-options", "'-fPIC'"]},
                     include_dirs=[numpy_include, CUDA["include"], "src"])

ext_cpu = Extension("_emc_cpu",
                    sources=["src/emc_cpu_swig_wrap.cpp",
                             "src/emc_cpu.cpp",
                             "src/calculate_responsabilities_cpu.cpp",
                             "src/calculate_scaling_cpu.cpp",
                             "src/update_slices_cpu.cpp"],
                    #extra_compile_args={"clang": []},
                    include_dirs=[numpy_include, "src"])

def customize_compiler_for_nvcc(self):
    self.src_extensions.append(".cu")
    default_compiler_so = self.compiler_so
    super = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        # print("\n\n\n")
        # print(extra_postargs)
        # print("\n\n\n")
        if os.path.splitext(src)[1] == ".cu":
            self.set_executable("compiler_so", CUDA["nvcc"])
            postargs = extra_postargs["nvcc"]            
        else:
            #self.set_executable("compiler_so", "/usr/bin/clang")
            #postargs = extra_postargs["clang"]
            postargs = []
            #pass
        super(obj, src, ext, cc_args, postargs, pp_opts)
        
        self.compiler_so = default_compiler_so

    self._compile = _compile
            

class custom_build_ext(build_ext):
    def build_extension(self, ext):
        customize_compiler_for_nvcc(self.compiler)
        #super(custom_build_ext, self).build_extension(ext)
        build_ext.build_extension(self, ext)

setuptools.setup(name="emc",
                 author="Tomas Ekeberg",
                 version="0.1",
                 py_modules=["emc", "emc_cuda", "emc_cpu"],
                 package_dir={"": "src"},
                 ext_modules=[ext_cuda, ext_cpu],
                 cmdclass={"build_ext": custom_build_ext},
                 zip_safe=False)

# setuptools.setup(name="emc",
#                  author="Tomas Ekeberg",
#                  version="0.1",
#                  py_modules=["emc", "emc_cpu"],
#                  package_dir={"": "src"},
#                  ext_modules=[ext_cpu],
#                  cmdclass={"build_ext": custom_build_ext},
#                  zip_safe=False)

