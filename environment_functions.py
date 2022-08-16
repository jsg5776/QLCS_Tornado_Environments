import numpy as np
#import numpy.ma as ma
import numpy.ma as ma
from metpy.units import units
import metpy.calc as mcalc
from metpy.interpolate import log_interpolate_1d
import time

####################################################################################
# Data Section
####################################################################################

## Sets up an NCSS query for RAP variables of interest, you can add other variables if needed
def query_data(ncss):
    query = ncss.query()
    query.all_times()
    query.variables('Convective_available_potential_energy_surface',
                        'Storm_relative_helicity_height_above_ground_layer',
                        'Convective_inhibition_surface',
                        'u-component_of_wind_isobaric',
                        'u-component_of_wind_height_above_ground',
                        'v-component_of_wind_isobaric',
                        'v-component_of_wind_height_above_ground',
                        'Composite_reflectivity_entire_atmosphere',
                        'Temperature_isobaric',
                        'Relative_humidity_isobaric', 
                        'Absolute_vorticity_isobaric', 
                        'Potential_temperature_height_above_ground', 
                        'Storm_relative_helicity_height_above_ground_layer', 
                        'Temperature_height_above_ground', 
                        'Precipitation_rate_surface')
    query.add_lonlat().lonlat_box(-135,-60,20,55) ## US box
    
    data = None 
    fn = 0
    
    while data is None:
        try: 
            if fn==6: 
                data = 'stop_count'
            data = ncss.get_data(query)
        except:
            print('didnt work')
            fn +=1
            pass
    print('worked')
    return data

## Finds nearest model(y,x) grid-point to a given lat lon value
def lat_lon_2D_index(y,x,lat1,lon1):
    import numpy as np
    '''This function calculates the distance from a desired 
    lat/lon point to each element of a 2D array of lat/lon values,
    typically from model output, and determines the index value 
    corresponding to the nearest lat/lon grid point.
    
    x = longitude array
    y = latitude array
    lon1 = longitude point (single value)
    lat1 = latitude point (single value)
    
    Returns the two index values for nearest lat, lon 
    point on grid for grids (y,x)
    
    Equations for variable distance between longitudes from
    http://andrew.hedges.name/experiments/haversine/'''
    
    R = 6373.*1000.  # Earth's Radius in meters
    rad = np.pi/180.
    x1 = np.ones(x.shape)*lon1
    y1 = np.ones(y.shape)*lat1
    dlon = np.abs(x-x1)
    dlat = np.abs(y-y1)
    a = (np.sin(rad*dlat/2.))**2 + \
         np.cos(rad*y1) * np.cos(rad*y) * (np.sin(rad*dlon/2.))**2 
    c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) ) 
    d = R * c
    return np.unravel_index(d.argmin(), d.shape)



####################################################################################
# Variable Calculation section (using metpy)
####################################################################################

## SRH Calculation for 2D model output
def calc_srh(u_p, v_p, z_p, p_profile, gpm_surface, p_surface, u_surface, v_surface, depth):
    
    srh = np.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))

    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0, len(gpm_surface[0,:])):
                surf = gpm_surface[ilat,ilon]
                gpm = z_p[:,ilat,ilon]-surf
                above_ground = np.where(gpm>units.Quantity(50,'m'))


                u = u_p[:,ilat,ilon][above_ground]
                v = v_p[:,ilat,ilon][above_ground]
                p = p_profile[above_ground]
                gpm = gpm[above_ground]

                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    u   = np.insert(u, 0, u_surface[ilat,ilon])
                    v   = np.insert(v, 0, v_surface[ilat,ilon])
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('srh')
                
                #print(gpm)

                [rm,_,mean] = mcalc.bunkers_storm_motion(units.Quantity(np.array(p), 'Pa'), units.Quantity(np.array(u),'m/s'),units.Quantity(np.array(v),'m/s'),units.Quantity(np.array(gpm),'gpm'))
                [srh_rm,srh_lm,tot] = mcalc.storm_relative_helicity(units.Quantity(np.array(gpm),'gpm'),  units.Quantity(np.array(u),'m/s'),units.Quantity(np.array(v),'m/s'), depth = units.Quantity(depth, 'm'), storm_u = rm[0], storm_v = rm[1])
                srh[ilat, ilon] = srh_rm.magnitude
    return srh
## Okubo-Weiss parameter
def ow(u,v, meters): 
    # Enter fields of u and v, and model grid spacing, returns OW parameter
    
    
    [dudy, dudx] = mcalc.gradient(u10, deltas = (units.Quantity(meters,'m'), units.Quantity(meters,'m')))
    [dvdy, dvdx] = mcalc.gradient(v10, deltas = (units.Quantity(meters,'m'), units.Quantity(meters,'m')))
    vort = mcalc.vorticity(u10, v10, dx = units.Quantity(meters,'m'), dy = units.Quantity(meters,'m'))
    
    
    d1 = dudx - dvdy
    d2 = dvdx + dudy
    
    
    w = d1.magnitude**2 + d2.magnitude**2 - vort.magnitude**2
    
    
    return w 

## mask based on MUCAPE values
def cape_mask(cape, precip, mucape): 
    # Enter previously subsetted fields of CAPE and Precip
    cape = ma.masked_where((precip > 0.001) | (mucape<50), cape)
    return cape

## LCL height over 2D model output
def calc_lcl_height(z_p, gpm_s, t_s, rh_s, p_s, p_profile): 
    
    td_s = mcalc.dewpoint_from_relative_humidity(t_s,  np.array(rh_s))

    lcl_p, lcl_t = mcalc.lcl(units.Quantity(np.array(p_s),'Pa'), units.Quantity(np.array(t_s), 'K'), units.Quantity(np.array(td_s),'degC'))

    lcl_height = ma.zeros((len(gpm_s[:,0]), len(gpm_s[0,:])))

    for ilat in range(0,len(gpm_s[:,0])):
        for ilon in range(0,len(gpm_s[0,:])):
                    surf_height = gpm_s[ilat,ilon]
                    gpm  = z_p[:,ilat,ilon]-surf_height
                    
                    #print(gpm)
                    above_ground = np.where(gpm>units.Quantity(50,'m'))
                    gpm = gpm[above_ground]
                    p_prof = p_profile[above_ground]
#                     print(p_prof)
#                     print(p_s[ilat,ilon])
                    
                    # append surface observation to the bottom of array
                    if p_s[ilat, ilon] > p_prof[0]: 
                        p_prof   = np.insert(p_prof, 0, p_s[ilat,ilon])
                        gpm = np.insert(0,0, gpm)
                    else: 
                        print('***Possible sounding error***')
                        print('lcl')
                        
                    #print(p_s[ilat,ilon]-p_prof[0])

                    lcl_height[ilat, ilon] = log_interpolate_1d(lcl_p[ilat,ilon].magnitude, p_prof.magnitude,  gpm.magnitude)
                    #print(lcl_height[ilat,ilon])
                    
#                     if np.isnan(lcl_height[ilat,ilon]):
#                         print(p_prof[0])
#                         print(p_s[ilat,ilon])
#                         print(lcl_p[ilat,ilon])
                    
    return lcl_height  #, td_s, t_s, gpm

## Bulk shear over 2D model output
def calc_bulk_shear(u_p, v_p, z_p, p_profile, gpm_surface, p_surface, u_surface, v_surface, depth):
    
    u_bulk = np.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    v_bulk = np.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    
    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0, len(gpm_surface[0,:])):
                surf = gpm_surface[ilat,ilon]
                gpm = z_p[:,ilat,ilon]-surf
                above_ground = np.where(gpm>units.Quantity(50,'m'))


                u = u_p[:,ilat,ilon][above_ground]
                v = v_p[:,ilat,ilon][above_ground]
                p = p_profile[above_ground]
                gpm = gpm[above_ground]
                
                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    u   = np.insert(u, 0, u_surface[ilat,ilon])
                    v   = np.insert(v, 0, v_surface[ilat,ilon])
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('blk')

                [u_bs, v_bs] = mcalc.bulk_shear(units.Quantity(np.array(p),'Pa'), units.Quantity(np.array(u), 'm/s'),\
                                                units.Quantity(np.array(v),'m/s'), height = units.Quantity(np.array(gpm),'gpm'),\
                                                depth=units.Quantity(depth,'m'))
                
                u_bulk[ilat, ilon] = u_bs.magnitude
                v_bulk[ilat, ilon] = v_bs.magnitude
                #print(u_bs)
    return u_bulk, v_bulk
## Most-unstable CAPE over 2D model output 

def calc_mucape(t_p, rh_p, z_p, p_profile, gpm_surface, p_surface, t_surface, rh_surface): 
    
    ## ****************************Should also be adding in a surface ob too **************************************
    MUCAPE = np.zeros((len(rh_p[0,:,0]), len(rh_p[0, 0,:])))
    MUCIN  = np.zeros((len(rh_p[0,:,0]), len(rh_p[0, 0,:])))

    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                surf_height = gpm_surface[ilat,ilon]
                
                gpm  = z_p[:,ilat,ilon]-surf_height
                above_ground = np.where(gpm>units.Quantity(50,'m'))

                # Above Ground 
                T  = t_p[:,ilat,ilon][above_ground]

                Td = mcalc.dewpoint_from_relative_humidity(T, np.array(rh_p[:,ilat,ilon][above_ground]))

                Td_surf = mcalc.dewpoint_from_relative_humidity(t_surface[ilat,ilon], np.array(rh_surface[ilat,ilon])) 
                p = p_profile[above_ground]
                gpm = gpm[above_ground]

                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    T   = np.insert(T, 0, t_surface[ilat,ilon])
                    Td  = np.insert(Td, 0, Td_surf)
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('mucape')
                [mucape, mucin]   = mcalc.most_unstable_cape_cin(units.Quantity(np.array(p), 'Pa'),\
                                                                 units.Quantity(np.array(T), 'K'),\
                                                                 units.Quantity(np.array(Td), 'degC'))
                
                MUCAPE[ilat,ilon] = mucape.magnitude
                MUCIN[ilat,ilon]  = mucin.magnitude

    return MUCAPE, MUCIN

## Calculate surface-based CAPE

def calc_sbcape(t_p, rh_p, z_p, p_profile, gpm_surface, p_surface, t_surface, rh_surface):
    
    ## Must feed in pint.Quantity because metPy thinks I can't handle units on my own
    
    start_init = time.time()
    sbCAPE = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    sbCIN  = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    end_init   = time.time()
    
    #print('init time = ', end_init-start_init)
    
    
    
    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                
                start_mask = time.time()
                surf_height = gpm_surface[ilat,ilon]
                gpm  = z_p[:,ilat,ilon]-surf_height
                above_ground = np.where(gpm>units.Quantity(50,'m'))
                end_mask = time.time()

                # Above Ground 
                
                Td_start = time.time()
                T  = t_p[:,ilat,ilon][above_ground]
                Td = mcalc.dewpoint_from_relative_humidity(T, np.array(rh_p[:,ilat,ilon][above_ground]))
                Td_surf = mcalc.dewpoint_from_relative_humidity(t_surface[ilat,ilon], np.array(rh_surface[ilat,ilon])) 
                Td_end   = time.time()
                
                #print('Td time = ', Td_end-Td_start)
                p = p_profile[above_ground]
                gpm = gpm[above_ground]

                # adding surface observations
                
                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    T   = np.insert(T, 0, t_surface[ilat,ilon])
                    Td  = np.insert(Td, 0, Td_surf)
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('sbcape')

                [sbcape, sbcin]   = mcalc.surface_based_cape_cin(units.Quantity(np.array(p), 'Pa'),\
                                                                 units.Quantity(np.array(T), 'K'),\
                                                                 units.Quantity(np.array(Td), 'degC'))
                #print('cape_time = ', end_cape-start_cape)
                sbCAPE[ilat,ilon] = sbcape.magnitude
                sbCIN[ilat,ilon]  = sbcin.magnitude

    return sbCAPE, sbCIN

## Calculate 0-3km CAPE
def calc_3kcape(t_p, rh_p, z_p, p_profile, gpm_surface, p_surface, t_surface, rh_surface):

    ## Must feed in pint.Quantity because metPy thinks I can't handle units on my own

    three_CAPE = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    three_CIN  = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))

    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                surf_height = gpm_surface[ilat,ilon]

                gpm  = z_p[:,ilat,ilon]-surf_height
                between = np.where((gpm>units.Quantity(50,'m')) & (gpm<units.Quantity(3000, 'm')))


                # Subsetting 
                T  = t_p[:,ilat,ilon][between]
                Td = mcalc.dewpoint_from_relative_humidity(T, np.array(rh_p[:,ilat,ilon][between]))

                Td_surf = mcalc.dewpoint_from_relative_humidity(t_surface[ilat,ilon], np.array(rh_surface[ilat,ilon])) 
                p = p_profile[between]
                gpm = gpm[between]

                # adding surface observations

                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    T   = np.insert(T, 0, t_surface[ilat,ilon])
                    Td  = np.insert(Td, 0, Td_surf)
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('3kcape')

                
                try:
                    [three_cape, three_cin]   = mcalc.surface_based_cape_cin(units.Quantity(np.array(p), 'Pa'),\
                                                                 units.Quantity(np.array(T), 'K'),\
                                                                 units.Quantity(np.array(Td), 'degC'))
                except: 
                    three_cape = units.Quantity(0, 'J/kg')
                    three_cin = units.Quantity(0, 'J/kg')
                    
                three_CAPE[ilat,ilon] = three_cape.magnitude
                three_CIN[ilat,ilon]  = three_cin.magnitude

    return three_CAPE, three_CIN
## 100 hPa (default) Mixed-layer CAPE 
def calc_mlcape(t_p, rh_p, z_p, p_profile, gpm_surface, p_surface, t_surface, rh_surface):
    
    ## Must feed in pint.Quantity because metPy thinks I can't handle units on my own
    
    mlCAPE = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    mlCIN  = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))

    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                surf_height = gpm_surface[ilat,ilon]
                
                gpm  = z_p[:,ilat,ilon]-surf_height
                above_ground = np.where(gpm>units.Quantity(50,'m'))

                # Above Ground 
                T  = t_p[:,ilat,ilon][above_ground]
                Td = mcalc.dewpoint_from_relative_humidity(T, np.array(rh_p[:,ilat,ilon][above_ground]))

                Td_surf = mcalc.dewpoint_from_relative_humidity(t_surface[ilat,ilon], np.array(rh_surface[ilat,ilon])) 
                p = p_profile[above_ground]
                gpm = gpm[above_ground]

# adding surface observations

                
                if p_surface[ilat, ilon] > p[0]: 
                    gpm = np.insert(gpm, 0,0)
                    T   = np.insert(T, 0, t_surface[ilat,ilon])
                    Td  = np.insert(Td, 0, Td_surf)
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('mlcape')
                
                [mlcape, mlcin]   = mcalc.mixed_layer_cape_cin(units.Quantity(np.array(p), 'Pa'),\
                                                                 units.Quantity(np.array(T), 'K'),\
                                                                 units.Quantity(np.array(Td), 'degC'))
#                 
                mlCAPE[ilat,ilon] = mlcape.magnitude
                mlCIN[ilat,ilon]  = mlcin.magnitude


    return mlCAPE, mlCIN

## Lapse rate between two layers (bot_height & height)
def calc_LR(t_p, z_p, p_profile, gpm_surface, t_surface, bot_height, height):
    ## Lapse rate tests

    LR  = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))

    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                
                surf_height = gpm_surface[ilat,ilon]

                gpm  = z_p[:,ilat,ilon]-surf_height
                #layer = np.where((gpm>units.Quantity(50,'m')) & (gpm<units.Quantity(height, 'm')))
                #t     = t_p[:,ilat,ilon]

                hgt = units.Quantity(height,'m')
                bhgt = units.Quantity(bot_height, 'm')

                km_temp = log_interpolate_1d(hgt, gpm, t_p[:,ilat,ilon])

                #p_test = p_profile[layer]
                #gpm = gpm[layer]
                #t   = t_test[:,ilat,ilon][layer]

                # adding surface observations

    #             gpm = np.insert(gpm, 0,0)
    #             T   = np.insert(t, 0, t_surface[ilat,ilon])
                if bot_height!=0: 
                    bot_temp = log_interpolate_1d(bhgt, gpm, t_p[:,ilat,ilon])
                else:
                    bot_temp = t_surface[ilat,ilon]
                LR[ilat, ilon] = (-(km_temp-bot_temp)/(height/1000)).magnitude
                #print(km_temp)
        
    return LR

## non-supercell tornado parameter 
def calc_nst(lr01, mlcape03, mlcin, bs06, vort): 
    mult1  = lr01/9 
    mult2  = mlcape03/100
    mult3  = (225-mlcin)/200
    mult4  = (18-bs06)/5
    mult5  = (vort/(8*10**-5))
    
    return mult1*mult2*mult3*mult4*mult5

## 0-3 km average RH
def calc_rh03(rh_p, rh_surf, p_prof, z_p, p_surface, gpm_surface): 
    rh03 = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    
    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                agl = z_p[:, ilat, ilon] - gpm_surface[ilat,ilon]
                
                above_ground = np.where(agl >units.Quantity(50,'m'))
                rh  = rh_p[:,ilat,ilon][above_ground]
                p = p_prof[above_ground]
                agl = agl[above_ground]
                
                                        
                ## Surface data
                
                if p_surface[ilat, ilon] > p[0]: 
                    agl = np.insert(agl, 0,0)
                    rh   = np.insert(rh, 0, rh_surf[ilat,ilon])
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('***Possible sounding error***')
                    print('rh03')
                
#                 rh_03 = mcalc.mean_pressure_weighted(units.Quantity(p, 'Pa'), rh, units.Quantity(agl,'m'), units.Quantity(0,'m')\
#                                                     , depth = units.Quantity(3000, 'm'))

                #rh_03 = mcalc.mean_pressure_weighted(units.Quantity(p, 'Pa'), np.array(rh))
                rh_03 = mcalc.mean_pressure_weighted(units.Quantity(np.array(p),'Pa'), units.Quantity(np.array(rh), 'm/m'), \
                                             height = agl, depth = units.Quantity(3000,'m'))
                rh03[ilat,ilon] = rh_03[0].magnitude
    return rh03

## 3-6 km avereage RH
def calc_rh36(rh_p, rh_surf, p_prof, z_p, p_surface, gpm_surface): 
    
    rh36 = ma.zeros((len(gpm_surface[:,0]), len(gpm_surface[0,:])))
    
    for ilat in range(0,len(gpm_surface[:,0])):
            for ilon in range(0,len(gpm_surface[0,:])):
                agl = z_p[:, ilat, ilon] - gpm_surface[ilat,ilon]
                
                above_ground = np.where(agl >units.Quantity(50,'m'))
                rh  = rh_p[:,ilat,ilon][above_ground]
                p = p_prof[above_ground]
                agl = agl[above_ground]
                
                
                ## Surface data
                if p_surface[ilat, ilon] > p[0]: 
                    agl = np.insert(agl, 0,0)
                    rh   = np.insert(rh, 0, rh_surf[ilat,ilon])
                    p   = np.insert(p, 0, p_surface[ilat,ilon])
                else: 
                    print('rh06')
                
#                 rh_03 = mcalc.mean_pressure_weighted(units.Quantity(p, 'Pa'), rh, units.Quantity(agl,'m'), units.Quantity(0,'m')\
#                                                     , depth = units.Quantity(3000, 'm'))

                #rh_03 = mcalc.mean_pressure_weighted(units.Quantity(p, 'Pa'), np.array(rh))
                rh_36 = mcalc.mean_pressure_weighted(units.Quantity(np.array(p),'Pa'), units.Quantity(np.array(rh), 'm/m'), \
                                             height = agl, bottom = units.Quantity(3000, 'm'), depth = units.Quantity(3000,'m'))
                rh36[ilat,ilon] = rh_36[0].magnitude
    return rh36

## Returns statistics and point value for a variable field at a given 2D index
def mean_median_max_pt(field, lat_indi, lon_indi):
    ## Given a 2D field
    
    field = ma.array(field)
    
    mn = ma.mean(field)
    md = ma.median(field)
    mx = ma.max(field)
    pt = field[lat_indi, lon_indi]

    return ma.array([mn,md,mx, pt])

## Returns statistics (with min instead of max) and point value for a variable field at a given 2D index
def mean_median_min_pt(field, lat_indi, lon_indi):
    ## Given a 2D field
    
    field = ma.array(field)
    
    mn = ma.mean(field)
    md = ma.median(field)
    mx = ma.min(field)
    pt = field[lat_indi, lon_indi]

    return ma.array([mn,md,mx, pt])
