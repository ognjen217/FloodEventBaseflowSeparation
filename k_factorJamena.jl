
using LinearAlgebra, Plots, DifferentialEquations
using Pkg, DataFrames, ForwardDiff
using DelimitedFiles, Optim
using LinearRegression
using GLM, CSV, LinearInterpolations
using LsqFit
using StatsBase
using Trapz
using DataInterpolations

# Define the Q data vector
Q = readdlm("D:\\DOCs, PDFs\\011 ExtremeClimTwin Project (2023)\\eventDataStations\\AprilMajJamenaQ.csv")

Q = vec(Q)
t = readdlm("D:\\DOCs, PDFs\\011 ExtremeClimTwin Project (2023)\\eventDataStations\\AprilMajJamenaT.csv")
t = vec(t)
# Compute the derivative of Q with respect to time using finite differences

interp_za_Q = LinearInterpolation(Q,t)
t = 0:0.0833:t[end]

Q = interp_za_Q.(t)

dQdt = diff(Q) ./ diff(t)
# Reshape the data for use with LinearRegression.jl
X = reshape(t[1:end-1], :, 1)
Y = reshape(dQdt, :, 1)

X = vec(X)
Y = vec(Y)
# Fit a linear model to the data
linReg = linregress(X, Y)

# Extract the slope of the linear model and compute the coefficient k
slope = coef(linReg)[1]
k = -1 / Q[1] * slope


####### estamination of model's accuracy ##########

Q_est = []
T = t

##for i in  1:712
##    pop!(Q_est)
##end
for i in 2:1:length(t)-1
    est = Q[i-1] * exp(k * (-1) * t[i] * 3)
    push!(Q_est, est)
end

plot(t[2:end-1], Q_est)
plot!(t,Q)
#print(Q_est./maximum(Q_est))
r2 = 1/2 .* (Q[2:end-1] .- Q_est).^2
mean(r2) / length(r2)

#r2_score(Q_est, Q)

r2_normalised = r2/maximum(r2)
r2_MA_threshold_value = 0.9
r2_filtered = r2[r2 .> r2_MA_threshold_value]
r2_plottable = (1 .- r2_normalised)

#=export=#

export_as_csv_file(t[2:end-1], r2_plottable, k_values_normalised, k, "r_kStar_k_jamena.csv")

#plot1 = plot( t, (Q ./ maximum(Q)), xlabel = " ", ylabel = " ", linecolor = :red, linewidth = 2, line=(:dashdot, 2), size=(1200,600), label = false)
plot1 = plot(t[2:end-1], [(Q ./ maximum(Q))[2:end-1], r2_plottable], xlabel = "t" ,ylabel = "r2", linewidth = 1.2, label=["Q" "r"])
plot1 = plot!(plot1, label=["Q" "r"])
plot!(legend=:upleft, legendcolumns=3)

#plot((Q_est ./ maximum(Q_est)))
t1 = t[2:end]
matx = [t1 Q_est[1:end-1]]
#plot(Q_est./maximum(Q_est))

savefig("myplot.png")

function find_k(Q::Vector{Float64})
    l = length(Q)
    k_values = Float64[]
    for t in eachindex(T)
        if t + 2 <= l
            dQdt = (Q[t+1] - Q[t]) / 3
            k = -1 / Q[t] * dQdt
            push!(k_values, k)
        end
    end
    return k_values
end

k_values = vec(find_k(vec(Q)))

function normalize_values(v::Vector{T}) where T<:Real
    # find minimum and maximum values
    min_value = -minimum(v)
    max_value = maximum(v)
    # normalize the values
    normalized_val = [];
    for i in eachindex(v) 
        if v[i] < 0
            v[i] = v[i] / min_value
        elseif v[i] > 0
            v[i] = v[i] / max_value
        end
    end    
    return v
end

k_values_normalised = normalize_values(k_values)
k_values_normalised = k_values ./ maximum(k_values)
ukupna_greska_racuna = 1/2 * sum((Q_est .- Q) .^ 2)
ukupna_greska_racuna_in_perc = ukupna_greska_racuna ./ sum(Q)

plot2 = plot(t[3:end], [(Q ./ maximum(Q))[3:end], k_values], xlabel = "t" ,ylabel = "k", linewidth = 1.2, label=["Q" "k"])
#plot2 = plot!(twinx(plot2),t, (Q ./ maximum(Q)), linecolor = :red, linewidth = 2.3, line=(:dashdot), size=(1200,600))


#recessing/rising points:
# =k*= :

k_star = change_rate_of_k_factor = (diff(vec(k_values_normalised)) ./ diff(vec(t[2:end-1])))
scatter(t[2:end-2], k_star, markersize=:0.6)
k_star_normalised = k_star./maximum(k_star)


k_star = find_k(k_values)

function select_data_points(k_values, k_star, r, Q, t)
    selected_time_points = Float64[]
    selected_Q_values = Float64[]

    for i in eachindex(k_star)
        if (-(k_values[i]) > 0 && abs(k_star[i])) < 3.6e-8 &&  r[i] < 0.001
            push!(selected_time_points, t[i])
            push!(selected_Q_values, Q[i])
        end

        if i-1 == 0
            continue
        elseif i+1==length(k_star)
            continue
        else
            if Q[i] < Q[i+1] 
                if Q[i] > Q[i+1]
                    push!(selected_Q_values, Q[i])
                    push!(selected_time_points, t[i])
                end
            end
        end
    end

    return selected_time_points, selected_Q_values
end

turn_pts_indx, turn_pts_Q = select_data_points(k_values, k_star, r2_normalised, Q[2:end-1], t[2:end-1])

scatter!(turn_pts_indx, turn_pts_Q, label="Vremenski trenuci")
#plot(turn_pts_indx, turn_pts_Q)

plot(t, Q, label = "Proticaj")
#plot!(twinx(),t[2:end-2], k_star)

savefig("passed_Q_points")
function find_local_maxima(Q, t, r2)
    maxima_values = Float64[]
    maxima_time_points = Float64[]
    
    for i in 2:length(Q)-1
        if  Q[i-1] < Q[i] 
            if Q[i] > Q[i+1]
                
                push!(maxima_values, Q[i])
                push!(maxima_time_points, t[i])
                
            end
        end
    end
    
    return maxima_values, maxima_time_points
end

maxima_values, maxima_time_points = find_local_maxima(Q, t, r2)
scatter(maxima_time_points, maxima_values, markersize = 1)
plot!(t,Q)

function find_local_minimum(Q, t)
    minima_values = Float64[]
    minima_time_points = Float64[]
    
    for i in 2:length(t)-1
        if Q[i] < Q[i-1]
            if Q[i] < Q[i+1] 
                push!(minima_values, Q[i])
                push!(minima_time_points, t[i])
            end 
        end
    end
    
    return minima_values, minima_time_points
end
minima_values, minima_time_points = find_local_minimum(Q, t)
scatter!(minima_time_points, minima_values,  markersize = 1)

plot3 = plot(t, Q ./ maximum(Q), linewidth = 1, linecolor = :red,line=(:dashdot, 1))
plot3 = plot(t[2:end-2], [(Q ./ maximum(Q))[2:end-2], change_rate_of_k_factor ./ maximum(change_rate_of_k_factor)], xlabel = "t" ,ylabel = "k*", linewidth = 1.2, label=["Q" "k*"])

#=plotter=#
plot3x1 = plot(plot1, plot2, plot3, layout=(3,1), size=(1200,800))
savefig("3x1plot.png")

scatter!(t,Q)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


#i)

function exp_fit(t, k)
    return Q[1] .* exp.(-k .* t)
end

# initial guess for k
k_guess = 0.9

# perform the fit
fit = curve_fit(exp_fit, t, Q, [k_guess])

# extract the fitted value of k
k_fit = fit.param[1]
J_fit = fit.jacobian
# print the fitted value of k
println("Fitted value of k = ", k_fit)

A = exp(-k_fit) #a value
k_star_t =  exp.(k .* t)
scatter(Q, abs.(dQdt), xlabel="Q", ylabel="dQdt")

#plot(dQdt) 
#plot!(Q, k_fit .* dqdQ) 

pl = plot(t, Q .* exp.(k_fit .* t), label="procenjeno Q na osnovu identif K")
plot!(twinx(pl), t, Q ./ maximum(Q), linewidth = 2, linecolor = :purple, line=(:dashdot, 1), label="Realno Q")

e = Q .- exp_fit(t, k)
r2_za_K = 1/2 .* (Q .- Q[1] .* exp.(k_fit .* t)).^2

#r2_score(Q_est, Q)

r2_normalised_za_K = r2_za_K/maximum(r2)
r2_MA_threshold_value_za_K = 0.95
r2_filtered_za_K = r2_za_K[r2_za_K .> r2_MA_threshold_value]
r2_plottable_za_K = 1 .- (1 .- r2_normalised_za_K)

pl1 = plot!(twinx(pl), t, r2_plottable_za_K./maximum(r2_plottable_za_K), linewidth = 2.1, linecolor = :grey, line=(:dot, 1), label = "greska procene", )
savefig(pl1, "QvsQestVSr2")

############################################
############################################
############################################
############################################
#III)) 
############################################
############################################
############################################
############################################
############################################


function find_maxima_and_fill_period(values, t)
    numb_of_values = length(values)
    maxima = []
    maxima_time_stamps = []
    Qm = Float64[]
    tb_te = Float64[]
    indexes = []

    for i in 2:numb_of_values-1
        if values[i] > values[i-1] && values[i] > values[i+1]
            push!(maxima, values[i])
            push!(maxima_time_stamps, t[i])
            push!(indexes, t[i])
        end
    end

    ##matrix_circuit_beaker_simulator
    
    te = (maxima_time_stamps[1] + maxima_time_stamps[2]) / 2
    tb = maxima_time_stamps[1] - te
    push!(Qm, maxima[1])
    push!(Qm, maxima[1])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(tb_te, te-eps(Float64))

    tb = te + eps(Float64)
    te = (maxima_time_stamps[2] + maxima_time_stamps[3]) / 2
    push!(Qm, maxima[2])
    push!(Qm, maxima[2])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))

    tb = te + eps(Float64)
    te = (maxima_time_stamps[3] + maxima_time_stamps[4]) / 2
    push!(Qm, maxima[3])
    push!(Qm, maxima[3])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))


    tb = te + eps(Float64)
    te = (maxima_time_stamps[4] + maxima_time_stamps[5]) / 2
    push!(Qm, maxima[4])
    push!(Qm, maxima[4])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))

    tb = te + eps(Float64)
    te = (maxima_time_stamps[5] + maxima_time_stamps[6]) / 2
    push!(Qm, maxima[5])
    push!(Qm, maxima[5])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))
    
    tb = te + eps(Float64)
    te = (maxima_time_stamps[6] + maxima_time_stamps[7]) / 2
    push!(Qm, maxima[6])
    push!(Qm, maxima[6])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))
    
    tb = te + eps(Float64)
    te = (maxima_time_stamps[7] + maxima_time_stamps[8]) / 2 
    push!(Qm, maxima[7])
    push!(Qm, maxima[7])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))
    
    tb = te + eps(Float64)
    te = (maxima_time_stamps[8] + maxima_time_stamps[9]) / 2 
    push!(Qm, maxima[8])
    push!(Qm, maxima[8])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))
 
    tb = te + eps(Float64)
    te = (maxima_time_stamps[9] + maxima_time_stamps[10]) / 2 
    push!(Qm, maxima[9])
    push!(Qm, maxima[9])
    print("izlaz",Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))
    
    tb = te + eps(Float64)
    te = (maxima_time_stamps[10] + maxima_time_stamps[10]-tb)
    push!(Qm, maxima[10])
    push!(Qm, maxima[10])
    print("izlaz", Qm)
    push!(tb_te, tb+eps(Float64))
    push!(indexes, tb)
    push!(tb_te, te-eps(Float64))

    return Qm, tb_te, indexes
end

matr = find_maxima_and_fill_period(Q, t)

Qn = Qm = vec(matr[1])
push!(Qn, minimum(Q))
tm = vec(matr[2])

tm[1] = 0
push!(tm, tm[length(tm)])
indexes = vec(matr[3 ])

pl1 = plot(t,Q);
plot!(tm, Qm)
#scatter!(tm, Q)
savefig("searcing_windows_jamena")
function calculate_bRDF(a, bfi, QQ, t)
    bRDF = Float64[] # Create an empty array to store the calculated bRDF values
    resize!(bRDF, length(QQ))
    # Calculate bRDF values for each time step
    for i = 1:length(t)-1
        if i == 1
            bRDF[i] = QQ[i]  # Initial value at time t = 1
        else
            bRDF[i] = (a * (1 - bfi)) / (1 - a * bfi) * bRDF[i-1] + ((1 - a) * bfi) / (1 - a * BFI) * QQ[i]
        end
    end

    println(bRDF)
    return bRDF

end

result = calculate_bRDF(A, BFI, Qn, tm)

int_za_Qn_za_passed_points = LinearInterpolation(t, Qn)

plot!(tm, result / maximum(result) .* Qn)
plot!(t, Q)
plot(t, BFI .* Q)

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

#odredjivanje funkcije raspodele#### CDF

Q_sorted = sort(Q)
itr = ecdf(Q_sorted)
n = length(Q_sorted)
mean_flow = mean(Q_sorted)

quantile(Q_sorted, 0.5; sorted=true)

bl = q_below = quantile(Q_sorted, 0.10)
bu = q_above = quantile(Q_sorted, 0.97)

function envelope_times_BFI()
    b_en = Float64[]
    for flowdata in Qn
        if (BFI * flowdata) >= bu
            push!(b_en, bu)
        elseif (BFI * flowdata) <= bl
            push!(b_en, bl)
        else
            push!(b_en, BFI * flowdata)
        end
    end

    return b_en
end

b_en = envelope_times_BFI()
b_enInterp = LinearInterpolation(b_en, tm)

t_cpy = tm
push!(t_cpy, t[end])
b_en_cpy = b_en
push!(b_en_cpy, Q[end])

b_en_interp = b_enInterp.(t)

plot!(t, min.(b_en_interp, Q))

plot!(tm, b_en) #nema turning points-a ispod baseflow nivoa
plot(t, Q)

#pretraga za turning points

b_en_inter_za_turning_points = b_enInterp.(turn_pts_indx)

scatter!(turn_pts_indx, b_en_inter_za_turning_points)

function check_lower_values(turn_pts_Q, b_en_inter_za_turning_points, turn_pts_indx)
    lower_values = []
    lower_time = []

    for i in eachindex(turn_pts_Q)
        if turn_pts_Q[i] < b_en_inter_za_turning_points[i]
            push!(lower_values, turn_pts_Q[i])
            push!(lower_time, turn_pts_indx[i])
        end
    end

    return  lower_time, lower_values
end


turn_pts_indx_passed, b_en_turning_points_passed = check_lower_values(turn_pts_Q, b_en_inter_za_turning_points, turn_pts_indx)

scatter!(turn_pts_indx_passed, b_en_turning_points_passed)

plot!(turn_pts_indx_passed, b_en_turning_points_passed, linewidth = 2, linecolor = :orange, line=(:dashdot, 1))

b_en_Turning_points_INTERP = LinearInterpolation(b_en_turning_points_passed, turn_pts_indx_passed)
b_en_Turning_points_interpolated = b_en_Turning_points_INTERP.(t)
b_en_Turning_points_below_FLOW = b_en_Turning_points_interpolated
plot!(t, b_en_Turning_points_below_FLOW, linewidth = 2.5, linecolor = :black, line=(:dot, 1))

plot!(t[40 .< t .< 120], Q[40 .< t .< 120], label = "Q", linecolor=:blue)

plot(t, Q)
b_rck = calculate_bRDF(A, BFI, b_en_inter_za_turning_points, turn_pts_indx)
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120],b_rck[40 .< turn_pts_indx .< 120], line=(:dashdot, 0.5),label = "Iteracija 1") 

b_rck1 = calculate_bRDF(A, BFI, b_rck, turn_pts_indx)
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120], b_rck1[40 .< turn_pts_indx .< 120],line=(:dashdot, 0.5), label = "Iteracija 2") 

b_rck2 = calculate_bRDF(A, BFI, b_rck1, turn_pts_indx)
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120], b_rck2[40 .< turn_pts_indx .< 120],line=(:dashdot, 0.5), label = "Iteracija 3") 

b_rck3 = calculate_bRDF(A, BFI, b_rck2, turn_pts_indx)
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120], b_rck3[40 .< turn_pts_indx .< 120],line=(:dashdot, 0.5), label = "Iteracija 4") 

b_rck4 = calculate_bRDF(A, BFI, b_rck3, turn_pts_indx)
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120], b_rck4[40 .< turn_pts_indx .< 120],line=(:dashdot, 0.5), label = "Iteracija 5") 

avg_bRCK4 = (b_rck1 .+ b_rck2 .+ b_rck3 .+ b_rck4) ./ 4
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120],avg_bRCK4[40 .< turn_pts_indx .< 120], label = "Prosek iteracija 1-5", line=(:dot, 4))

savefig("EventSeparated1-5_jamena")
#plot

#b_rck5 = calculate_bRDF(A, BFI, b_rck4, turn_pts_indx)
#plot!(turn_pts_indx,b_rck5) 

#plot!(turn_pts_indx,b_rck6)

brckQ = calculate_bRDF(A, BFI, BFI*Q, t)
brckQ1 = calculate_bRDF(A, BFI, brckQ, t)
brckQ2 = calculate_bRDF(A, BFI, brckQ1, t)
brckQ3 = calculate_bRDF(A, BFI, brckQ2, t)
brckQ4 = calculate_bRDF(A, BFI, brckQ3, t)
brckQ5 = calculate_bRDF(A, BFI, brckQ4, t)

plot!(t, brckQ)
plot!(t[1:end-50], brckQ1[1:end-50])
plot!(t[1:end-50], brckQ2[1:end-50])
plot!(t[1:end-50], brckQ3[1:end-50])
plot!(t[1:end-50], brckQ4[1:end-50])
plot!(t[1:end-50], brckQ5[1:end-50])

avg_brckQ =  brckQ[1:end-20] .+ brckQ1[1:end-20] .+  brckQ2[1:end-20] .+ brckQ3[1:end-20] .+ brckQ4[1:end-20] .+ brckQ5[1:end-20]
plot!(turn_pts_indx[40 .< turn_pts_indx .< 120], avg_bRCK4[40 .< turn_pts_indx .< 120], label = "Prosek iteracija 1-5", line=(:dot, 4))
plot!(t[1:end-20], avg_brckQ ./ 6, line=(:dot, 4))

savefig("eckhart_vs_ognjen_vs_mei_jamena")


export_as_csv_file(turn_pts_indx, b_rck, b_rck1, b_rck2, b_rck3, "jamena_brck.csv")
export_as_csv_file(turn_pts_indx, turn_pts_Q, "jamena_points_passed_075.csv")

export_as_csv_file(t, brckQ, brckQ1, brckQ2, brckQ3, "jamenaEckhartMethod.csv")

######################################################
######################################################
######################################################
######################################################
######################################################

function export_as_csv_file(vectort, vector0, vector1, vector2, vector3, filename)
    data = DataFrame(Vector = vectort, VectorQ = interp_za_Q.(turn_pts_indx), Vector0 = vector0, Vector1 = vector1,
                     Vector2 = vector2, Vector3 = vector3)
    CSV.write(filename, data)
end

function export_as_csv_file(vectort, vector0, vector1, vector2, vector3, filename)
    data = DataFrame(VectorT = vectort, VectorQ = BFI .* Q, VectorEkh1 = vector0, VectorEkh2 = vector1,
                 VectorEkh3 = vector2, VectorEkh4 = vector3)
    CSV.write(filename, data)
end

function export_as_csv_file(vectort, vector0, filename)
    data = DataFrame(T_for_passed_points_high_res = vectort, Q_for_passed_points_high_res = vector0)
    CSV.write(filename, data)
end

function export_as_csv_file(vectort, vectorR, vectorK_Star, vectorK, filename)
    data = DataFrame(T = vectort, vecR = vectorR, vecKstar = vectorK_Star, vecK = vectorK)
    CSV.write(filename, data)
end

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

##pronalazenje BFI

######################################
#####################################

function find_local_minima(data::Vector{T}, time_points::Vector{S}) where {T, S}
    
    n = length(data)
    minima_values = Vector{T}()
    minima_indices = Vector{Int}()

    for i in 3:5:n-2
        subvector = data[i-2:i+2]
        min_index = argmin(subvector)

        push!(minima_values, subvector[min_index])
        push!(minima_indices, i-2+min_index)
    end

    return minima_values, minima_indices
end

function filter_data(data::Vector{Float64}, time_points::Vector{Int64})
    n = length(data)
    filtered_data = Vector{Float64}()
    filtered_time_points = Vector{Float64}()

    for i in 2:n-1
        if 0.9 * data[i] < min(data[i-1], data[i+1])
            push!(filtered_data, data[i])
            push!(filtered_time_points, time_points[i])
        end
    end

    return filtered_data, filtered_time_points
end
######################################

mitrovicaBASE = readdlm("D:\\DOCs, PDFs\\011 ExtremeClimTwin Project (2023)\\JuliaDevelopment\\jamenaSVE.csv")
t_indx = []

for i in eachindex(mitrovicaBASE)
    push!(t_indx, i)
end

local_minima_with_time_values = find_local_minima(vec(mitrovicaBASE), vec(t_indx))
values_of_minima              = local_minima_with_time_values[1] 
values_of_time                = local_minima_with_time_values[2]
filtered_data                 = filter_data(values_of_minima, values_of_time)
filtered_base                 = filtered_data[1]
filtered_base_time            = filtered_data[2]  
tttt = 1:1:length(mitrovicaBASE)
t_indx

plot(filtered_base_time[ 200 .< filtered_base_time .< 500], filtered_base[200 .<filtered_base_time.<500])
plot(tttt[1650 .<tttt .< 2200], mitrovicaBASE[1650 .<tttt .< 2200], label = "Proticaj")

#Qb 
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################
########################################################


interp = LinearInterpolation(filtered_base, filtered_base_time)
filtered_base_interpolated = interp.(t_indx)


# Find the minimum value at each time point
min_values_of_flow_BASEEEEE = [min(val1, val2) for (val1, val2) in zip(filtered_base_interpolated, mitrovicaBASE)]

plot!(t_indx[1650 .< t_indx .< 2200], min_values_of_flow_BASEEEEE[1650 .< t_indx .< 2200], label = "UKIH")
plot!(t_indx, filtered_base_interpolated)
savefig("UKIH_1650-2200_jamena")
sum(min_values_of_flow_BASEEEEE ./ mitrovicaBASE) / length(mitrovicaBASE)

mitrovicaBASE = vec(mitrovicaBASE)
t_indx_vec = Float64[]
for i in eachindex(t_indx) 
    push!(t_indx_vec, t_indx[i])
end

flow_interp_for_integr = LinearInterpolation(mitrovicaBASE, t_indx_vec)

mitrovicaBASE_high_freq = flow_interp_for_integr.(1:0.001:length(t_indx_vec))
baseflow_high_freq = interp.(1:0.001:length(t_indx_vec))

BFI = sum(baseflow_high_freq ./ mitrovicaBASE_high_freq) / length(mitrovicaBASE_high_freq)

#########################################
#########################################
#########################################
#########################################
#
brckQ = calculate_bRDF(A, BFI, BFI * Q, t)
brckQ1 = calculate_bRDF(A, BFI, brckQ, t)
brckQ2 = calculate_bRDF(A, BFI, brckQ1, t)
brckQ3 = calculate_bRDF(A, BFI, brckQ2, t)
brckQ4 = calculate_bRDF(A, BFI, brckQ3, t)
brckQ5 = calculate_bRDF(A, BFI, brckQ4, t)


plot!(t, brckQ)
plot!(t[1:end-20], brckQ1[1:end-20])
plot!(t[1:end-20], brckQ2[1:end-20])
plot!(t[1:end-20], brckQ3[1:end-20])
plot!(t[1:end-20], brckQ4[1:end-20])
plot!(t[1:end-20], brckQ5[1:end-20])

avg_brckQ =  brckQ[1:end-20] .+ brckQ1[1:end-20] .+  brckQ2[1:end-20] .+ brckQ3[1:end-20] .+ brckQ4[1:end-20] .+ brckQ5[1:end-20]
plot!(t[1:end-20], avg_brckQ ./ 6)

savefig("eckhart vs ognjen vs mei")

