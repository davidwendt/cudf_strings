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
    // we could pass in and check the type too (optional parameter)
    DataHandler( PyObject* obj, const char* validate_type=nullptr )
    : pyobj(obj), values(0), count(0), type_width(0)
    {
        if( pyobj == Py_None )
            return; // not an error (e.g. nulls bitmask)

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
        if( errortext.empty() && validate_type &&
            (dtype_name.compare(validate_type)!=0) )
        {
            errortext = "argument must be of type ";
            errortext += validate_type;
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

// from custr's cpp/src/utilities/type_dispatcher.hpp
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
        T* items = reinterpret_cast<T*>(data);
        auto result = new custr::category<T>(items,count);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};

struct base_functor
{
    custr::base_type* obj_ptr;
    base_functor(custr::base_type* obj_ptr) : obj_ptr(obj_ptr) {}
};
}

//
static PyObject* n_createCategoryFromBuffer( PyObject* self, PyObject* args )
{
    PyObject* pyobj = PyTuple_GetItem(args,0); // only one parm expected
    DataHandler data(pyobj);
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"category.create: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( data.get_dtype_name(), create_functor(data.get_values(),data.get_count()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr(result);
}

// called by destructor in python class
static PyObject* n_destroyCategory( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    Py_BEGIN_ALLOW_THREADS
    printf("calling category dtor(%p)\n",tptr);
    delete tptr;
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

namespace {
struct size_functor : base_functor
{
    size_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (reinterpret_cast<custr::category<T>*>(obj_ptr))->size(); }
};
}
static PyObject* n_size( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    size_t count;
    Py_BEGIN_ALLOW_THREADS
    count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(count);
}

namespace {
struct keys_size_functor : base_functor
{
    keys_size_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> size_t operator()() { return (reinterpret_cast<custr::category<T>*>(obj_ptr))->keys_size(); }
};
}
static PyObject* n_keys_size( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    size_t count;
    Py_BEGIN_ALLOW_THREADS
    count = type_dispatcher( tptr->get_type_name(), keys_size_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(count);
}

namespace {
struct get_keys_functor : base_functor
{
    get_keys_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    void operator()(void* keys)
    {
        custr::category<T>* this_ptr = reinterpret_cast<custr::category<T>*>(obj_ptr);
        memcpy(keys,this_ptr->keys(),this_ptr->keys_size()*sizeof(T));
    }
};
}
static PyObject* n_get_keys( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    DataHandler data(PyTuple_GetItem(args,1),tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_keys: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), keys_size_functor(tptr) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", count, tptr->get_type_name());
        Py_RETURN_NONE;
    }
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), get_keys_functor(tptr), data.get_values() );
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

namespace {
struct get_values_functor : base_functor
{
    get_values_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    void operator()(void* values)
    {
        custr::category<T>* this_ptr = reinterpret_cast<custr::category<T>*>(obj_ptr);
        memcpy(values,this_ptr->values(),this_ptr->size()*sizeof(int)); //!!!!!!
    }
};
}
static PyObject* n_get_values( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    DataHandler data_handler(PyTuple_GetItem(args,1),"int32");
    if( data_handler.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_values: %s", data_handler.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    if( count > data_handler.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld int32 values", count);
        Py_RETURN_NONE;
    }
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), get_values_functor(tptr), data_handler.get_values() );
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

// this one cannot use a functor since the passed in key must be converted from python
static PyObject* n_get_indexes_for_key( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykey = PyTuple_GetItem(args,1);
    PyObject* pyoutput = PyTuple_GetItem(args,2);
    if( pyoutput == Py_None )
        Py_RETURN_NONE;

    DataHandler data(pyoutput,"int32");
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_indexes_for_key: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    std::string dtype = tptr->get_type_name();
    int* results = reinterpret_cast<int*>(data.get_values());
    size_t count = 0;
    if( dtype.compare("int32")==0 )
    {
        int key = (int)PyLong_AsLong(pykey);
        Py_BEGIN_ALLOW_THREADS
        custr::category<int>* obj = reinterpret_cast<custr::category<int>*>(tptr);
        count = obj->get_indexes_for(key,results);
        Py_END_ALLOW_THREADS
    }
    else if( dtype.compare("int64")==0 )
    {
        long key = (long)PyLong_AsLong(pykey);
        Py_BEGIN_ALLOW_THREADS
        custr::category<long>* obj = reinterpret_cast<custr::category<long>*>(tptr);
        count = obj->get_indexes_for(key,results);
        Py_END_ALLOW_THREADS
    }
    else if( dtype.compare("float32")==0 )
    {
        float key =  (float)PyFloat_AsDouble(pykey);
        Py_BEGIN_ALLOW_THREADS
        custr::category<float>* obj = reinterpret_cast<custr::category<float>*>(tptr);
        count = obj->get_indexes_for(key,results);
        Py_END_ALLOW_THREADS
    }
    else if( dtype.compare("float64")==0 )
    {
        double key = (double)PyFloat_AsDouble(pykey);
        Py_BEGIN_ALLOW_THREADS
        custr::category<double>* obj = reinterpret_cast<custr::category<double>*>(tptr);
        count = obj->get_indexes_for(key,results);
        Py_END_ALLOW_THREADS
    }
    else
    {
        throw std::runtime_error("invalid dtype in category<> get_indexes");
    }

    //
    return PyLong_FromLong(count);
}

namespace {
struct get_indexes_for_null_functor : base_functor
{
    get_indexes_for_null_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T>
    size_t operator()(int* result)
    {
        custr::category<T>* this_ptr = reinterpret_cast<custr::category<T>*>(obj_ptr);
        return this_ptr->get_indexes_for_null_key(result);
    }
};
}
static PyObject* n_get_indexes_for_null_key( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    DataHandler data(PyTuple_GetItem(args,1),"int32");
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"get_indexes_for_null: %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld int32 values", count);
        Py_RETURN_NONE;
    }
    Py_BEGIN_ALLOW_THREADS
    count = type_dispatcher( tptr->get_type_name(), get_indexes_for_null_functor(tptr), reinterpret_cast<int*>(data.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromLong(count);
}

namespace {
struct to_type_functor : base_functor
{
    to_type_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(void* items, BYTE* nulls)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        T* results = reinterpret_cast<T*>(items);
        cthis->to_type(results,nulls);
    }
};
}
static PyObject* n_to_type( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pydata = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler data(pydata,tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"to_type %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", count, tptr->get_type_name() );
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), to_type_functor(tptr), data.get_values(), reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

namespace {
struct gather_type_functor : base_functor
{
    gather_type_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void operator()(int* indexes, unsigned int count, void* items, BYTE* nulls)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        T* results = reinterpret_cast<T*>(items);
        cthis->gather_type(indexes,count,results,nulls);
    }
};
}
static PyObject* n_gather_type( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    PyObject* pydata = PyTuple_GetItem(args,2);
    PyObject* pynulls = PyTuple_GetItem(args,3);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"indexes %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler data(pydata,tptr->get_type_name());
    if( data.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"output %s", data.get_error_text());
        Py_RETURN_NONE;
    }
    size_t count = type_dispatcher( tptr->get_type_name(), size_functor(tptr) );
    if( count > data.get_count() )
    {
        PyErr_Format(PyExc_ValueError,"buffer must be able to hold at least %ld %s values", count, tptr->get_type_name() );
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    Py_BEGIN_ALLOW_THREADS
    type_dispatcher( tptr->get_type_name(), gather_type_functor(tptr),
                     reinterpret_cast<int*>(indexes.get_values()), indexes.get_count(),
                     data.get_values(), reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}

namespace {
struct gather_functor : base_functor
{
    gather_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->gather(indexes,count);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_gather( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), indexes.get_count());
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr(result);
}

namespace {
struct gather_values_functor : base_functor
{
    gather_values_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->gather_values(indexes,count);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_gather_values( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather_values %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_values_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), indexes.get_count());
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct gather_remap_functor : base_functor
{
    gather_remap_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(int* indexes, unsigned int count)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->gather_and_remap(indexes,count);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_gather_and_remap( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pyindexes = PyTuple_GetItem(args,1);
    DataHandler indexes(pyindexes,"int32");
    if( indexes.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"gather_and_remap %s", indexes.get_error_text());
        Py_RETURN_NONE;
    }
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), gather_remap_functor(tptr),
                              reinterpret_cast<int*>(indexes.get_values()), indexes.get_count());
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct add_keys_functor : base_functor
{
    add_keys_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->add_keys(reinterpret_cast<T*>(keys),count,nulls);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_add_keys( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"add_keys: indexes %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), add_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct remove_keys_functor : base_functor
{
    remove_keys_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->remove_keys(reinterpret_cast<T*>(keys),count,nulls);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_remove_keys( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"remove_keys: %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), remove_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct remove_unused_functor : base_functor
{
    remove_unused_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()()
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->remove_unused_keys();
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_remove_unused( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), remove_unused_functor(tptr) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct set_keys_functor : base_functor
{
    set_keys_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* keys, unsigned int count, BYTE* nulls)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        auto result = cthis->set_keys(reinterpret_cast<T*>(keys),count,nulls);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_set_keys( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pykeys = PyTuple_GetItem(args,1);
    PyObject* pynulls = PyTuple_GetItem(args,2);
    DataHandler keys(pykeys,tptr->get_type_name());
    if( keys.is_error() )
    {
        PyErr_Format(PyExc_ValueError,"set_keys: %s", keys.get_error_text());
        Py_RETURN_NONE;
    }
    DataHandler nulls(pynulls);
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), set_keys_functor(tptr),
                              keys.get_values(), keys.get_count(),
                              reinterpret_cast<BYTE*>(nulls.get_values()) );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
}

namespace {
struct merge_functor : base_functor
{
    merge_functor(custr::base_type* obj_ptr) : base_functor(obj_ptr) {}
    template<typename T> void* operator()(void* cat)
    {
        custr::category<T>* cthis = reinterpret_cast<custr::category<T>*>(obj_ptr);
        custr::category<T>* cthat = reinterpret_cast<custr::category<T>*>(cat);
        auto result = cthis->merge(*cthat);
        result->print();
        return reinterpret_cast<void*>(result);
    }
};
}
static PyObject* n_merge_category( PyObject* self, PyObject* args )
{
    custr::base_type* tptr = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(PyTuple_GetItem(args,0)));
    PyObject* pycat = PyTuple_GetItem(args,1);
    if( pycat == Py_None )
    {
        PyErr_Format(PyExc_ValueError,"merge: argument cannot be null");
        Py_RETURN_NONE;
    }
    // also could check pycat type is category type
    custr::base_type* tcat = reinterpret_cast<custr::base_type*>(PyLong_AsVoidPtr(pycat));
    void* result;
    Py_BEGIN_ALLOW_THREADS
    result = type_dispatcher( tptr->get_type_name(), merge_functor(tptr), tcat );
    Py_END_ALLOW_THREADS
    return PyLong_FromVoidPtr((void*)result);
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
    { "n_get_indexes_for_null_key", n_get_indexes_for_null_key, METH_VARARGS, "" },
    { "n_to_type", n_to_type, METH_VARARGS, "" },
    { "n_gather_type", n_gather_type, METH_VARARGS, "" },
    { "n_merge_category", n_merge_category, METH_VARARGS, "" },
    { "n_add_keys", n_add_keys, METH_VARARGS, "" },
    { "n_remove_keys", n_remove_keys, METH_VARARGS, "" },
    { "n_remove_unused", n_remove_unused, METH_VARARGS, "" },
    { "n_set_keys", n_set_keys, METH_VARARGS, "" },
    { "n_gather", n_gather, METH_VARARGS, "" },
    { "n_gather_values", n_gather_values, METH_VARARGS, "" },
    { "n_gather_and_remap", n_gather_and_remap, METH_VARARGS, "" },
    { NULL, NULL, 0, NULL }
};

static struct PyModuleDef cModPyDem = {	PyModuleDef_HEAD_INIT, "cucategory_module", "", -1, s_Methods };

PyMODINIT_FUNC PyInit_pynicucategory(void)
{
    return PyModule_Create(&cModPyDem);
}

