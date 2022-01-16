using Tables, DataFrames


# based on https://github.com/TheRoniOne/MDTable.jl/blob/master/src/write.jl
function write_mdtable(io::IO, df)
    rows = Tables.rows(df)
    sch = Tables.schema(rows)
    names = Tables.columnnames(rows)
    # header = true
    headers::String = ""
    for i in 1:length(names)
        if i != length(names)
            headers = headers * "| $(names[i]) "
        else
            headers = headers * "| $(names[i]) " * "|\n"
        end
    end
    write(io, headers)

    write(io, "| --- " ^ length(names) * "|\n")
    for row in rows
        line::String = ""
        Tables.eachcolumn(sch, row) do val, i, nm
            line = line * "| $val "
        end
        write(io, line * "|\n")
    end

end