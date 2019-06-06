#include <Python.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <stdexcept>
#include "include/category.h"

//
namespace {
class DataHandler
{
    PyObject* pyobj;
    std::string errortext;
    unsigned int type_width;
    std::string dtype_name;
    void* values;
    unsigned int count;

public:
    //
    DataHandler( PyObject* obj ) : pyobj(obj), values(0), count(0), type_width(0)
    {
        if( pyobj == Py_None )
        {
            errortext = "null object";
            return;
        }

        std::string name = pyobj->ob_type->tp_name;
        if( name.compare("DeviceNDArray")==0 )
        {
            PyObject* pyasize = PyObject_GetAttr(pyobj,PyUnicode_FromString("alloc_size"));
            PyObject* pysize = PyObject_GetAttr(pyobj,PyUnicode_FromString("size"));
            PyObject* pydtype = PyObject_GetAttr(pyobj,PyUnicode_FromString("dtype"));
            PyObject* pydcp = PyObject_GetAttr(pyobj,PyUnicode_FromString("device_ctypes_pointer"));
            pyobj = PyObject_GetAttr(pydcp,PyUnicode_FromString("value"));
            //printf("dnda: size=%d, alloc_size=%d\n",(int)PyLong_AsLong(pysize),(int)PyLong_AsLong(pyasize));
            count = (unsigned int)PyLong_AsLong(pysize);
            if( count > 0 )
                type_width = PyLong_AsLong(pyasize)/count;
            //printf("dnda: count=%d, twidth=%d\n",(int)count,(int)type_width);
            if( pyobj != Py_None )
            {
                values = PyLong_AsVoidPtr(pyobj);
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else if( name.compare("numpy.ndarray")==0 )
        {
            PyObject* pyasize = PyObject_GetAttr(pyobj,PyUnicode_FromString("nbytes"));
            PyObject* pysize = PyObject_GetAttr(pyobj,PyUnicode_FromString("size"));
            PyObject* pydtype = PyObject_GetAttr(pyobj,PyUnicode_FromString("dtype"));
            PyObject* pydcp = PyObject_GetAttr(pyobj,PyUnicode_FromString("ctypes"));
            pyobj = PyObject_GetAttr(pydcp,PyUnicode_FromString("data"));
            //printf("nda: size=%d, alloc_size=%d\n",(int)PyLong_AsLong(pysize),(int)PyLong_AsLong(pyasize));
            count = (unsigned int)PyLong_AsLong(pysize);
            if( count > 0 )
                type_width = PyLong_AsLong(pyasize)/count;
            //printf("nda: count=%d, twidth=%d\n",(int)count,(int)type_width);
            if( pyobj != Py_None )
            {
                values = PyLong_AsVoidPtr(pyobj);
                dtype_name = PyUnicode_AsUTF8(PyObject_Str(pydtype));
            }
        }
        else
        {
            errortext = "unknown_type: ";
            errortext += name;
        }
    }

    //
    ~DataHandler()
    {}

    //
    bool is_error()               { return !errortext.empty(); }
    const char* get_error_text()  { return errortext.c_str(); }

    void* get_values()            { return values; }
    unsigned int get_count()      { return count; }
    unsigned int get_type_width() { return type_width; }
    const char* get_dtype_name()  { return dtype_name.c_str(); }
};

// from cudf's cpp/src/utilities/type_dispatcher.hpp
template<class functor_t, typename... Ts>
constexpr decltype(auto) type_dispatcher( const char* stype, functor_t fn, Ts&&... args )
{
    std::string dtype = stype;
    if( dtype.compare("int32")==0 )
        return fn.template operator()<int>(std::forward<Ts>(args)...);
    if( dtype.compare("int64")==0 )
        return fn.template operator()<long>(std::forward<Ts>(args)...);
    if( dtype.compare("float32")==0 )
        return fn.template operator()<float>(std::forward<Ts>(args)...);
    if( dtype.compare("float64")==0 )
        return fn.template operator()<double>(std::forward<Ts>(args)...);
    //
    throw std::runtime_error("invalid dtype in category<> dispatcher");
}

//
struct category_wrapper
{
    std::string dtype;
    void* this_pointer;
    category_wrapper( const char* stype, void* data ) : dtype(stype), this_pointer(data) {}
};
}

namespace {
struct create_functor
{
    void* data;
    size_t count;
    create_functor(void* data, size_t count) : data(data), count(count) {}
    template<typename T>
    void* operator()()
    {
        T* items = static_cast<T*>(data);
        printf("calling category ctor(%p,%ld)\n",items,count);
        auto result = new cudf::category<T>(items,count);
        result->print();
        return static_cast<void*>(result);
    }
};

struct base_functor
{
    void* obj_ptr;
    base_functor(void* obj_ptr) : obj_ptr(obj_ptr) {}
};
}

//
static PyObject* n_createCategoryFromBuffer( PyObject* self, PyObject* args )
{
    PyObject* pyobj = PyTuple_GetItem(args,0); // only one parm expected
    if( pyobj == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: parameter required");
        Py_RETURN_NONE;
    }

    DataHandler data_handler(pyobj);
    if( data_handler.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: %s", data_handler.get_error_text());
        Py_RETURN_NONE;
    }

    std::string dtype = data_handler.get_dtype_name();
    void* result = type_dispatcher( dtype.c_str(), create_functor(data_handler.get_values(),data_handler.get_count()) );
    category_wrapper* rtn = new category_wrapper(dtype.c_str(),result);
    return PyLong_FromVoidPtr((void*)rtn);
}

namespace {
struct delete_functor : base_functor
{
    delete_functor( void* obj_ptr ) : base_functor(obj_ptr) {}
    template<typename T> void operator()() 
    {
        printf("calling category dtor(%p)\n",obj_ptr);
        delete static_cast<T*>(obj_ptr);
    }
};
}
// called by destructor in python class
static PyObject* n_destroyCategory( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    type_dispatcher( tptr->dtype.c_str(), delete_functor(tptr->this_pointer) );
    delete tptr;
    return PyLong_FromLong(0);
}

namespace {
struct size_functor : base_functor
{
    size_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (static_cast<cudf::category<T>*>(obj_ptr))->size(); }
};
}
static PyObject* n_size( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    size_t count = type_dispatcher( tptr->dtype.c_str(), size_functor(tptr->this_pointer) );
    return PyLong_FromLong(count);
}

namespace {
struct keys_size_functor : base_functor
{
    keys_size_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (static_cast<cudf::category<T>*>(obj_ptr))->keys_size(); }
};
}
static PyObject* n_keys_size( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    size_t count = type_dispatcher( tptr->dtype.c_str(), keys_size_functor(tptr->this_pointer) );
    return PyLong_FromLong(count);
}

namespace {
struct get_keys_functor : base_functor
{
    get_keys_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> 
    void operator()(void* keys)
    {
        cudf::category<T>* this_ptr = static_cast<cudf::category<T>*>(obj_ptr);
        memcpy(keys,this_ptr->keys(),this_ptr->keys_size()*sizeof(T));
    }
};
}
static PyObject* n_get_keys( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    DataHandler data_handler(PyTuple_GetItem(args,1));
    if( data_handler.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: %s", data_handler.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = tptr->dtype;
    if( dtype.compare(data_handler.get_dtype_name())!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be %s", dtype.c_str());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( dtype.c_str(), keys_size_functor(tptr->this_pointer) );
    if( count > data_handler.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be at least %ld", count);
        Py_RETURN_NONE;
    }
    type_dispatcher( dtype.c_str(), get_keys_functor(tptr->this_pointer), data_handler.get_values() );
    Py_RETURN_NONE;
}

namespace {
struct get_values_functor : base_functor
{
    get_values_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> 
    void operator()(void* values)
    {
        cudf::category<T>* this_ptr = static_cast<cudf::category<T>*>(obj_ptr);
        memcpy(values,this_ptr->values(),this_ptr->size()*sizeof(int));
    }
};
}
static PyObject* n_get_values( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    DataHandler data_handler(PyTuple_GetItem(args,1));
    if( data_handler.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: %s", data_handler.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = data_handler.get_dtype_name();
    if( dtype.compare("int32")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be int32");
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->dtype.c_str(), size_functor(tptr->this_pointer) );
    if( count > data_handler.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be at least %ld", count);
        Py_RETURN_NONE;
    }
    type_dispatcher( dtype.c_str(), get_values_functor(tptr->this_pointer), data_handler.get_values() );
    Py_RETURN_NONE;
}

//
static PyObject* n_get_indexes_for_key( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    size_t count = type_dispatcher( tptr->dtype.c_str(), size_functor(tptr->this_pointer) );
    PyObject* pykey = PyTuple_GetItem(args,1);
    PyObject* pyoutput = PyTuple_GetItem(args,2);
    if( pyoutput == Py_None )
        return PyLong_FromLong(count);

    DataHandler data_handler(pyoutput);
    if( data_handler.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: %s", data_handler.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = data_handler.get_dtype_name();
    if( dtype.compare("int32")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be int32");
        Py_RETURN_NONE;
    }
    dtype = tptr->dtype;
    int* results = static_cast<int*>(data_handler.get_values());
    if( dtype.compare("int32")==0 )
    {
        int key = (int)PyLong_AsLong(pykey);
        cudf::category<int>* obj = static_cast<cudf::category<int>*>(tptr->this_pointer);
        count = obj->get_indexes_for(key,results);
    }
    else if( dtype.compare("int64")==0 )
    {
        long key = (long)PyLong_AsLong(pykey);
        cudf::category<long>* obj = static_cast<cudf::category<long>*>(tptr->this_pointer);
        count = obj->get_indexes_for(key,results);
    }
    else if( dtype.compare("float32")==0 )
    {
        float key =  (float)PyFloat_AsDouble(pykey);
        cudf::category<float>* obj = static_cast<cudf::category<float>*>(tptr->this_pointer);
        count = obj->get_indexes_for(key,results);
    }
    else if( dtype.compare("float64")==0 )
    {
        double key = (double)PyFloat_AsDouble(pykey);
        cudf::category<double>* obj = static_cast<cudf::category<double>*>(tptr->this_pointer);
        count = obj->get_indexes_for(key,results);
    }
    else
    {
        throw std::runtime_error("invalid dtype in category<> get_indexes");
    }
    
    //
    return PyLong_FromLong(count);
}

namespace {
struct to_type_functor : base_functor
{
    to_type_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(void* items, BYTE* nulls)
    {
        cudf::category<T>* cthis = static_cast<cudf::category<T>*>(obj_ptr);
        T* results = static_cast<T*>(items);
        cthis->to_type(results,nulls);
    }
};
}
static PyObject* n_to_type( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pydata = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler data(pydata);
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = tptr->dtype;
    if( dtype.compare(data.get_dtype_name())!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be %s", dtype.c_str());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( dtype.c_str(), size_functor(tptr->this_pointer) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be at least %ld", count);
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    type_dispatcher( dtype.c_str(), to_type_functor(tptr->this_pointer), data.get_values(), static_cast<BYTE*>(nulls.get_values()) );

    Py_RETURN_NONE;
}

namespace {
struct gather_type_functor : base_functor
{
    gather_type_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(int* indexes, unsigned int count, void* items, BYTE* nulls)
    {
        cudf::category<T>* cthis = static_cast<cudf::category<T>*>(obj_ptr);
        T* results = static_cast<T*>(items);
        cthis->gather_type(indexes,count,results,nulls);
    }
};
}
static PyObject* n_gather_type( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    PyObject* pydata = PyTuple_GetItem(args,2);
    PyObject* pynulls = PyTuple_GetItem(args,3);
    DataHandler indexes(pyindexes);
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = indexes.get_dtype_name();
    if( dtype.compare("int32")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes buffer must be int32");
        Py_RETURN_NONE;
    }
    DataHandler data(pydata);
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    dtype = tptr->dtype;
    if( dtype.compare(data.get_dtype_name())!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be %s", dtype.c_str());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( dtype.c_str(), size_functor(tptr->this_pointer) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: buffer must be at least %ld", count);
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    type_dispatcher( dtype.c_str(), gather_type_functor(tptr->this_pointer), 
                     static_cast<int*>(indexes.get_values()), indexes.get_count(), 
                     data.get_values(), static_cast<BYTE*>(nulls.get_values()) );

    Py_RETURN_NONE;
}

namespace {
struct gather_functor : base_functor
{
    gather_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        cudf::category<T>* cthis = static_cast<cudf::category<T>*>(obj_ptr);
        auto result = cthis->gather(indexes,count);
        result->print();
        return static_cast<void*>(result);
    }
};
}
static PyObject* n_gather( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes);
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = indexes.get_dtype_name();
    if( dtype.compare("int32")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes buffer must be int32");
        Py_RETURN_NONE;
    }
    void* result = type_dispatcher( dtype.c_str(), gather_functor(tptr->this_pointer), 
                                    static_cast<int*>(indexes.get_values()), indexes.get_count());
    category_wrapper* rtn = new category_wrapper(dtype.c_str(),result);
    return PyLong_FromVoidPtr((void*)rtn);
}

namespace {
struct gather_values_functor : base_functor
{
    gather_values_functor(void* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        cudf::category<T>* cthis = static_cast<cudf::category<T>*>(obj_ptr);
        auto result = cthis->gather_values(indexes,count);
        result->print();
        return static_cast<void*>(result);
    }
};
}
static PyObject* n_gather_values( PyObject* self, PyObject* args )
{
    category_wrapper* tptr = (category_wrapper*)PyLong_AsVoidPtr(PyTuple_GetItem(args,0));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes);
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = indexes.get_dtype_name();
    if( dtype.compare("int32")!=0 )
    {
        PyErr_Format(PyExc_ValueError,"nvcategory: indexes buffer must be int32");
        Py_RETURN_NONE;
    }
    void* result = type_dispatcher( dtype.c_str(), gather_values_functor(tptr->this_pointer), 
                                    static_cast<int*>(indexes.get_values()), indexes.get_count());
    category_wrapper* rtn = new category_wrapper(dtype.c_str(),result);
    return PyLong_FromVoidPtr((void*)rtn);
}

//
static PyMethodDef s_Methods[] = {
    { "n_createCategoryFromBuffer", n_createCategoryFromBuffer, METH_VARARGS, "" },
    { "n_destroyCategory", n_destroyCategory, METH_VARARGS, "" },
    { "n_size", n_size, METH_VARARGS, "" },
    { "n_keys_size", n_keys_size, METH_VARARGS, "" },
    { "n_get_keys", n_get_keys, METH_VARARGS, "" },
    { "n_get_values", n_get_values, METH_VARARGS, "" },
    { "n_get_indexes_for_key", n_get_indexes_for_key, METH_VARARGS, "" },
    { "n_to_type", n_to_type, METH_VARARGS, "" },
    { "n_gather_type", n_gather_type, METH_VARARGS, "" },
//    { "n_merge_category", n_merge_category, METH_VARARGS, "" },
//    { "n_add_keys", n_add_keys, METH_VARARGS, "" },
//    { "n_remove_keys", n_remove_keys, METH_VARARGS, "" },
//    { "n_remove_unused", n_remove_unused, METH_VARARGS, "" },
//    { "n_set_keys", n_set_keys, METH_VARARGS, "" },
    { "n_gather", n_gather, METH_VARARGS, "" },
    { "n_gather_values", n_gather_values, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "cucategory_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pynicucategory(void)
{
    return PyModule_Create(&cModPyDem);
}

