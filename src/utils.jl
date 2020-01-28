abstract type DataSet end

struct TrainDataSet <: DataSet
    rootdir::AbstractString
end

struct ValDataSet <: DataSet
    rootdir::AbstractString
end
