
function rrmarcrossval(mardata, p, r̄=[], a=1, b=1, maxiters=1000, tucketa=1e-04, ϵ=1e-01, train=0.7)
    N1, N2, obs = size(mardata)
    if r̄ == []
        r̄ = [N1, N2, N1, N2]
    end
    grid = collect(Iterators.product(1:r̄[1], 1:r̄[2], 1:r̄[3], 1:r̄[4]))
    traintotal = Int(round(obs * train)) - 1
    testtotal = obs - traintotal - 1
    cvest = fill(NaN, 5, prod(r̄))

    # Model Selection
    Threads.@threads for i in ProgressBar(1:prod(r̄))
        selectedrank = collect(grid[i])
        r1, r2, r3, r4 = selectedrank
        if r1 > r2 * r3 * r4 || r2 > r1 * r3 * r4 || r3 > r1 * r2 * r4 || r4 > r1 * r2 * r3
            cvest[2, i] = r1
            cvest[3, i] = r2
            cvest[4, i] = r3
            cvest[5, i] = r4
        else
            # time series cross validation
            errortable = fill(NaN, testtotal)
            for j in 1:testtotal
                traindata = mardata[:, :, j:(traintotal+j)]
                testdata = mardata[:, :, traintotal+j+1]
                tuckest = tuckerreg(traindata, selectedrank; tucketa, a, b, maxiters, p, ϵ)
                tuckpred = contract(tuckest.A[:, :, :, 1:9], [3, 4], traindata[:, :, end], [1, 2]) + contract(tuckest.A[:, :, :, 10:18], [3, 4], traindata[:, :, end-1], [1, 2])
                # Root Mean Squared Error
                errortable[j] = norm(tuckpred - testdata)
            end
            cvest[1, i] = mean(errortable .^ 2)
            cvest[2, i] = r1
            cvest[3, i] = r2
            cvest[4, i] = r3
            cvest[5, i] = r4
        end
        GC.gc()
    end
    nancols = findall(x -> any(isnan, x), eachcol(cvest))
    filteredic = cvest[:, setdiff(1:size(cvest, 2), nancols)]
    cvvec = argmin(filteredic[1, :])
    cvchosen = Int.(filteredic[2:end, cvvec])

    return (CV=cvchosen, ictable=cvest)
end
