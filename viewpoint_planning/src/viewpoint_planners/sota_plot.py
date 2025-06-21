import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

threshold = 0.1
min_occlusion = 0.01
Gray_star_size=20
# Create the plot
fig, ax = plt.subplots()
G3_gsnbv_linewidth = 0.5
True_line_alpha = 0.15
terminate_size = 200


G3_scnbvp_linewidth = 0.5
G3_scnbvp1= [(False, 0.3608), (False, 0.25), (False, 0.2889), (False, 0.2955), 
             (False, 0.2889), (True, 0.0174),
  (True, 0.0174),(True, 0.0174),(True, 0.0174),(True, 0.0174),(True, 0.0174)  ]
colors_G3_scnbvp1 = ['cyan' if not value[0] else 'grey' for value in G3_scnbvp1]
x1_G3_scnbvp1 = list(range(len(G3_scnbvp1)))
y1_G3_scnbvp1 = [1 - value[1] for value in G3_scnbvp1]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x1_G3_scnbvp1)-1):
    if G3_scnbvp1[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x1_G3_scnbvp1[i:i+2], y1_G3_scnbvp1[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x1_G3_scnbvp1[i:i+2], y1_G3_scnbvp1[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x1_G3_scnbvp1[i:i+2], y1_G3_scnbvp1[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp1[i])  # Normal line
        
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x1_G3_scnbvp1)):
    if G3_scnbvp1[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x1_G3_scnbvp1[i], y1_G3_scnbvp1[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp1[i][0]:  # Other True points in grey
        ax.scatter(x1_G3_scnbvp1[i], y1_G3_scnbvp1[i], color='cyan', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x1_G3_scnbvp1[i], y1_G3_scnbvp1[i], color=colors_G3_scnbvp1[i], marker='x',  s=80)

G3_scnbvp2= [(False, 0.832), (False, 0.3868), (False, 0.4324), (False, 0.01), 
             (False, 0.2214), (False, 0.2273), (True, 0.0404),
  (True, 0.0404),(True, 0.0404),(True, 0.0404),(True, 0.0404)  ]
colors_G3_scnbvp2 = ['cyan' if not value[0] else 'grey' for value in G3_scnbvp2]
x2_G3_scnbvp2 = list(range(len(G3_scnbvp2)))
y2_G3_scnbvp2 = [1 - value[1] for value in G3_scnbvp2]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x2_G3_scnbvp2)-1):
    if G3_scnbvp2[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x2_G3_scnbvp2[i:i+2], y2_G3_scnbvp2[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x2_G3_scnbvp2[i:i+2], y2_G3_scnbvp2[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x2_G3_scnbvp2[i:i+2], y2_G3_scnbvp2[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp2[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x2_G3_scnbvp2)):
    if G3_scnbvp2[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x2_G3_scnbvp2[i], y2_G3_scnbvp2[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp2[i][0]:  # Other True points in grey
        ax.scatter(x2_G3_scnbvp2[i], y2_G3_scnbvp2[i], color='cyan', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x2_G3_scnbvp2[i], y2_G3_scnbvp2[i], color=colors_G3_scnbvp2[i], marker='x',  s=80)

G3_scnbvp3= [(False, 0.2277), (False, 0.2217), (False, 0.048), (True, 0.0327), 
             (True, 0.0327), (True, 0.0327), (True, 0.0327), (True, 0.0327),(True, 0.0327),(True, 0.0327),(True, 0.0327)  ]
colors_G3_scnbvp3 = ['cyan' if not value[0] else 'grey' for value in G3_scnbvp3]
x3_G3_scnbvp3 = list(range(len(G3_scnbvp3)))
y3_G3_scnbvp3 = [1 - value[1] for value in G3_scnbvp3]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x3_G3_scnbvp3)-1):
    if G3_scnbvp3[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x3_G3_scnbvp3[i:i+2], y3_G3_scnbvp3[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x3_G3_scnbvp3[i:i+2], y3_G3_scnbvp3[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x3_G3_scnbvp3[i:i+2], y3_G3_scnbvp3[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp3[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x3_G3_scnbvp3)):
    if G3_scnbvp3[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x3_G3_scnbvp3[i], y3_G3_scnbvp3[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp3[i][0]:  # Other True points in grey
        ax.scatter(x3_G3_scnbvp3[i], y3_G3_scnbvp3[i], color='cyan', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x3_G3_scnbvp3[i], y3_G3_scnbvp3[i], color=colors_G3_scnbvp3[i], marker='x',  s=80)

G3_scnbvp4= [(False, 0.2111), (True, 0.116), (False, 0.4474), (True, 0.0577), 
             (True, 0.0577), (True, 0.0577), (True, 0.0577), (True, 0.0577),(True, 0.0577),(True, 0.0577),(True, 0.0577)  ]
colors_G3_scnbvp4 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp4]
x4_G3_scnbvp4 = list(range(len(G3_scnbvp4)))
y4_G3_scnbvp4 = [1 - value[1] for value in G3_scnbvp4]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x4_G3_scnbvp4)-1):
    if G3_scnbvp4[i][0] and not first_true_found and  G3_scnbvp4[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x4_G3_scnbvp4[i:i+2], y4_G3_scnbvp4[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x4_G3_scnbvp4[i:i+2], y4_G3_scnbvp4[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x4_G3_scnbvp4[i:i+2], y4_G3_scnbvp4[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp4[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x4_G3_scnbvp4)):
    if G3_scnbvp4[i][0] and not first_true_found and  G3_scnbvp4[i][1] <threshold:  # First True point in blue
        ax.scatter(x4_G3_scnbvp4[i], y4_G3_scnbvp4[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp4[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x4_G3_scnbvp4[i], y4_G3_scnbvp4[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp4[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x4_G3_scnbvp4[i], y4_G3_scnbvp4[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x4_G3_scnbvp4[i], y4_G3_scnbvp4[i], color=colors_G3_scnbvp4[i], marker='x',  s=80)

G3_scnbvp5= [(False, 0.2402), (False, 0.01), (False, 0.4565), (True, 0.4565), 
             (True, 0.0119), (True, 0.0119), (True, 0.0119), (True, 0.0119),(True, 0.0119),(True, 0.0119),(True, 0.0119)  ]
colors_G3_scnbvp5 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp5]
x5_G3_scnbvp5 = list(range(len(G3_scnbvp5)))
y5_G3_scnbvp5 = [1 - value[1] for value in G3_scnbvp5]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x5_G3_scnbvp5)-1):
    if G3_scnbvp5[i][0] and not first_true_found and  G3_scnbvp5[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x5_G3_scnbvp5[i:i+2], y5_G3_scnbvp5[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x5_G3_scnbvp5[i:i+2], y5_G3_scnbvp5[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x5_G3_scnbvp5[i:i+2], y5_G3_scnbvp5[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp5[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x5_G3_scnbvp5)):
    if G3_scnbvp5[i][0] and not first_true_found and  G3_scnbvp5[i][1] <threshold:  # First True point in blue
        ax.scatter(x5_G3_scnbvp5[i], y5_G3_scnbvp5[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp5[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x5_G3_scnbvp5[i], y5_G3_scnbvp5[i], color='cyan', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x5_G3_scnbvp5[i], y5_G3_scnbvp5[i], color=colors_G3_scnbvp5[i], marker='x',  s=80)


G3_scnbvp6= [(False, 0.1765), (False, 0.4737), (False, 0.0215), (False, 0.3881), 
             (False, 0.3571), (False, 0.2517), (False, 0.424), 
             (False, 0.424), (False, 0.3715), (False, 0.3715), (False, 1.0)]
colors_G3_scnbvp6 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp6]
x6_G3_scnbvp6 = list(range(len(G3_scnbvp6)))
y6_G3_scnbvp6 = [1 - value[1] for value in G3_scnbvp6]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x6_G3_scnbvp6)-1):
    if G3_scnbvp6[i][0] and not first_true_found and  G3_scnbvp6[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x6_G3_scnbvp6[i:i+2], y6_G3_scnbvp6[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x6_G3_scnbvp6[i:i+2], y6_G3_scnbvp6[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x6_G3_scnbvp6[i:i+2], y6_G3_scnbvp6[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp6[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x6_G3_scnbvp6)):
    if G3_scnbvp6[i][0] and not first_true_found and  G3_scnbvp6[i][1] <threshold:  # First True point in blue
        ax.scatter(x6_G3_scnbvp6[i], y6_G3_scnbvp6[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp6[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x6_G3_scnbvp6[i], y6_G3_scnbvp6[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp6[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x6_G3_scnbvp6[i], y6_G3_scnbvp6[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x6_G3_scnbvp6[i], y6_G3_scnbvp6[i], color=colors_G3_scnbvp6[i], marker='x',  s=80)


G3_scnbvp7= [(False, 0.225), (False, 0.4492), (False, 0.0599), (False, 0.4916), 
             (False, 0.4873), (False, 0.4135), (False, 0.4135),
             (False, 0.419), (False, 0.7212), (False, 0.7212), (False, 0.2392)]
colors_G3_scnbvp7 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp7]
x7_G3_scnbvp7 = list(range(len(G3_scnbvp7)))
y7_G3_scnbvp7 = [1 - value[1] for value in G3_scnbvp7]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x7_G3_scnbvp7)-1):
    if G3_scnbvp7[i][0] and not first_true_found and  G3_scnbvp7[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x7_G3_scnbvp7[i:i+2], y7_G3_scnbvp7[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x7_G3_scnbvp7[i:i+2], y7_G3_scnbvp7[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x7_G3_scnbvp7[i:i+2], y7_G3_scnbvp7[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp7[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x7_G3_scnbvp7)):
    if G3_scnbvp7[i][0] and not first_true_found and  G3_scnbvp7[i][1] <threshold:  # First True point in blue
        ax.scatter(x7_G3_scnbvp7[i], y7_G3_scnbvp7[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp7[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x7_G3_scnbvp7[i], y7_G3_scnbvp7[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp7[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x7_G3_scnbvp7[i], y7_G3_scnbvp7[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x7_G3_scnbvp7[i], y7_G3_scnbvp7[i], color=colors_G3_scnbvp7[i], marker='x',  s=80)
        

   
G3_scnbvp8 =[(False, 0.2222), (False, 0.2139), (False, 0.4267), (False, 0.4267), 
             (False, 0.4267), (False, 0.4327),
 (False, 0.4206), (False, min_occlusion), (False, min_occlusion), (False, 1.0), 
 (False, 1.0)]     
colors_G3_scnbvp8 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp8]
x8_G3_scnbvp8 = list(range(len(G3_scnbvp8)))
y8_G3_scnbvp8 = [1 - value[1] for value in G3_scnbvp8]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x8_G3_scnbvp8)-1):
    if G3_scnbvp8[i][0] and not first_true_found and  G3_scnbvp8[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x8_G3_scnbvp8[i:i+2], y8_G3_scnbvp8[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x8_G3_scnbvp8[i:i+2], y8_G3_scnbvp8[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x8_G3_scnbvp8[i:i+2], y8_G3_scnbvp8[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp8[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x8_G3_scnbvp8)):
    if G3_scnbvp8[i][0] and not first_true_found and  G3_scnbvp8[i][1] <threshold:  # First True point in blue
        ax.scatter(x8_G3_scnbvp8[i], y8_G3_scnbvp8[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp8[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x8_G3_scnbvp8[i], y8_G3_scnbvp8[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp8[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x8_G3_scnbvp8[i], y8_G3_scnbvp8[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x8_G3_scnbvp8[i], y8_G3_scnbvp8[i], color=colors_G3_scnbvp8[i], marker='x',  s=80)        
        


G3_scnbvp9 =[(False, 0.2111), (False, 1.0), (False, 0.2167), (False, 0.1967),
  (False, 0.2131), (False, 0.2167), (True, 0.062) ,(True, 0.062), 
  (True, 0.062), (True, 0.062), (True, 0.062)]     
colors_G3_scnbvp9 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp9]
x9_G3_scnbvp9 = list(range(len(G3_scnbvp9)))
y9_G3_scnbvp9 = [1 - value[1] for value in G3_scnbvp9]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x9_G3_scnbvp9)-1):
    if G3_scnbvp9[i][0] and not first_true_found and  G3_scnbvp9[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x9_G3_scnbvp9[i:i+2], y9_G3_scnbvp9[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x9_G3_scnbvp9[i:i+2], y9_G3_scnbvp9[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x9_G3_scnbvp9[i:i+2], y9_G3_scnbvp9[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp9[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x9_G3_scnbvp9)):
    if G3_scnbvp9[i][0] and not first_true_found and  G3_scnbvp9[i][1] <threshold:  # First True point in blue
        ax.scatter(x9_G3_scnbvp9[i], y9_G3_scnbvp9[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp9[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x9_G3_scnbvp9[i], y9_G3_scnbvp9[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp9[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x9_G3_scnbvp9[i], y9_G3_scnbvp9[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x9_G3_scnbvp9[i], y9_G3_scnbvp9[i], color=colors_G3_scnbvp9[i], marker='x',  s=80)          
        
        
G3_scnbvp10 =[(False, 0.2227), (False, 0.1679), (True, 0.1214), (False, 0.1687),
              (True, 0.309), (False, 0.2566), 
 (False, 0.4043), (False, 0.4222), (False, 0.0615), (False, 0.2317), 
 (False, 1.0)]
colors_G3_scnbvp10 = ['cyan' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_scnbvp10]
x10_G3_scnbvp10 = list(range(len(G3_scnbvp10)))
y10_G3_scnbvp10 = [1 - value[1] for value in G3_scnbvp10]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x10_G3_scnbvp10)-1):
    if G3_scnbvp10[i][0] and not first_true_found and  G3_scnbvp10[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x10_G3_scnbvp10[i:i+2], y10_G3_scnbvp10[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x10_G3_scnbvp10[i:i+2], y10_G3_scnbvp10[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x10_G3_scnbvp10[i:i+2], y10_G3_scnbvp10[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_scnbvp10[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x10_G3_scnbvp10)):
    if G3_scnbvp10[i][0] and not first_true_found and  G3_scnbvp10[i][1] <threshold:  # First True point in blue
        ax.scatter(x10_G3_scnbvp10[i], y10_G3_scnbvp10[i], color='cyan', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp10[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x10_G3_scnbvp10[i], y10_G3_scnbvp10[i], color='cyan', marker='*',  s=Gray_star_size)
    elif G3_scnbvp10[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x10_G3_scnbvp10[i], y10_G3_scnbvp10[i], color='cyan', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x10_G3_scnbvp10[i], y10_G3_scnbvp10[i], color=colors_G3_scnbvp10[i], marker='x',  s=80)          
        






G3_gsnbv1= [(False, 0.1829), (False, 1), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208)]
colors_G3_gsnbv = ['red' if not value[0] else 'grey' for value in G3_gsnbv1]
x1_G3_gsnbv1 = list(range(len(G3_gsnbv1)))
y1_G3_gsnbv1 = [1 - value[1] for value in G3_gsnbv1]
# Plot line data for Line 1
first_true_found = False
for i in range(len(x1_G3_gsnbv1)-1):
    if G3_gsnbv1[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x1_G3_gsnbv1[i:i+2], y1_G3_gsnbv1[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x1_G3_gsnbv1[i:i+2], y1_G3_gsnbv1[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x1_G3_gsnbv1[i:i+2], y1_G3_gsnbv1[i:i+2], linewidth=G3_gsnbv_linewidth,color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x1_G3_gsnbv1)):
    if G3_gsnbv1[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x1_G3_gsnbv1[i], y1_G3_gsnbv1[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv1[i][0]:  # Other True points in grey
        ax.scatter(x1_G3_gsnbv1[i], y1_G3_gsnbv1[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x1_G3_gsnbv1[i], y1_G3_gsnbv1[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv2= [(False, 0.1829), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021), (True, 0.021)]
x2_G3_gsnbv2 = list(range(len(G3_gsnbv2)))
y2_G3_gsnbv2 = [1 - value[1] for value in G3_gsnbv2]
first_true_found = False
for i in range(len(x2_G3_gsnbv2)-1):
    if G3_gsnbv2[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x2_G3_gsnbv2[i:i+2], y2_G3_gsnbv2[i:i+2], color='gray', linestyle='--',linewidth=G3_gsnbv_linewidth, alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x2_G3_gsnbv2[i:i+2], y2_G3_gsnbv2[i:i+2], color='gray', linestyle='--',linewidth=G3_gsnbv_linewidth, alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x2_G3_gsnbv2[i:i+2], y2_G3_gsnbv2[i:i+2], color=colors_G3_gsnbv[i],linewidth=G3_gsnbv_linewidth)  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x2_G3_gsnbv2)):
    if G3_gsnbv2[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x2_G3_gsnbv2[i], y2_G3_gsnbv2[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv2[i][0]:  # Other True points in grey
        ax.scatter(x2_G3_gsnbv2[i], y2_G3_gsnbv2[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x2_G3_gsnbv2[i], y2_G3_gsnbv2[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv3= [(False, 0.2151), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019)]
x3_G3_gsnbv3 = list(range(len(G3_gsnbv3)))
y3_G3_gsnbv3 = [1 - value[1] for value in G3_gsnbv3]
first_true_found = False
for i in range(len(x3_G3_gsnbv3)-1):
    if G3_gsnbv3[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x3_G3_gsnbv3[i:i+2], y3_G3_gsnbv3[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x3_G3_gsnbv3[i:i+2], y3_G3_gsnbv3[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth,linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x3_G3_gsnbv3[i:i+2], y3_G3_gsnbv3[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x3_G3_gsnbv3)):
    if G3_gsnbv3[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x3_G3_gsnbv3[i], y3_G3_gsnbv3[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv3[i][0]:  # Other True points in grey
        ax.scatter(x3_G3_gsnbv3[i], y3_G3_gsnbv3[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x3_G3_gsnbv3[i], y3_G3_gsnbv3[i], color=colors_G3_gsnbv[i], marker='x',  s=80)


G3_gsnbv4= [(False, 0.2126), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019)]
x4_G3_gsnbv4 = list(range(len(G3_gsnbv4)))
y4_G3_gsnbv4 = [1 - value[1] for value in G3_gsnbv4]
first_true_found = False
for i in range(len(x4_G3_gsnbv4)-1):
    if G3_gsnbv4[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x4_G3_gsnbv4[i:i+2], y4_G3_gsnbv4[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x4_G3_gsnbv4[i:i+2], y4_G3_gsnbv4[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x4_G3_gsnbv4[i:i+2], y4_G3_gsnbv4[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x4_G3_gsnbv4)):
    if G3_gsnbv4[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x4_G3_gsnbv4[i], y4_G3_gsnbv4[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv4[i][0]:  # Other True points in grey
        ax.scatter(x4_G3_gsnbv4[i], y4_G3_gsnbv4[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x4_G3_gsnbv4[i], y4_G3_gsnbv4[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv5= [(False, 0.1914), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211), (True, 0.0211)]
x5_G3_gsnbv5 = list(range(len(G3_gsnbv5)))
y5_G3_gsnbv5 = [1 - value[1] for value in G3_gsnbv5]
first_true_found = False
for i in range(len(x5_G3_gsnbv5)-1):
    if G3_gsnbv5[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x5_G3_gsnbv5[i:i+2], y5_G3_gsnbv5[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x5_G3_gsnbv5[i:i+2], y5_G3_gsnbv5[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x5_G3_gsnbv5[i:i+2], y5_G3_gsnbv5[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x5_G3_gsnbv5)):
    if G3_gsnbv5[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x5_G3_gsnbv5[i], y5_G3_gsnbv5[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv5[i][0]:  # Other True points in grey
        ax.scatter(x5_G3_gsnbv5[i], y5_G3_gsnbv5[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x5_G3_gsnbv5[i], y5_G3_gsnbv5[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv6= [(False, 0.1709), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208), (True, 0.0208)]
x6_G3_gsnbv6 = list(range(len(G3_gsnbv6)))
y6_G3_gsnbv6 = [1 - value[1] for value in G3_gsnbv6]
first_true_found = False
for i in range(len(x6_G3_gsnbv6)-1):
    if G3_gsnbv6[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x6_G3_gsnbv6[i:i+2], y6_G3_gsnbv6[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x6_G3_gsnbv6[i:i+2], y6_G3_gsnbv6[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x6_G3_gsnbv6[i:i+2], y6_G3_gsnbv6[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x5_G3_gsnbv5)):
    if G3_gsnbv6[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x5_G3_gsnbv5[i], y6_G3_gsnbv6[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv6[i][0]:  # Other True points in grey
        ax.scatter(x5_G3_gsnbv5[i], y6_G3_gsnbv6[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x5_G3_gsnbv5[i], y6_G3_gsnbv6[i], color=colors_G3_gsnbv[i], marker='x',  s=80)


G3_gsnbv7= [(False, 0.3853), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184), (True, 0.0184)]
x7_G3_gsnbv7 = list(range(len(G3_gsnbv7)))
y7_G3_gsnbv7 = [1 - value[1] for value in G3_gsnbv7]
first_true_found = False
for i in range(len(x7_G3_gsnbv7)-1):
    if G3_gsnbv7[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x7_G3_gsnbv7[i:i+2], y7_G3_gsnbv7[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x7_G3_gsnbv7[i:i+2], y7_G3_gsnbv7[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x7_G3_gsnbv7[i:i+2], y7_G3_gsnbv7[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x7_G3_gsnbv7)):
    if G3_gsnbv7[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x7_G3_gsnbv7[i], y7_G3_gsnbv7[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv7[i][0]:  # Other True points in grey
        ax.scatter(x7_G3_gsnbv7[i], y7_G3_gsnbv7[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x7_G3_gsnbv7[i], y7_G3_gsnbv7[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv8= [(False, 0.2105), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152), (True, 0.0152)]
x8_G3_gsnbv8 = list(range(len(G3_gsnbv8)))
y8_G3_gsnbv8 = [1 - value[1] for value in G3_gsnbv8]
first_true_found = False
for i in range(len(x8_G3_gsnbv8)-1):
    if G3_gsnbv8[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x8_G3_gsnbv8[i:i+2], y8_G3_gsnbv8[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x8_G3_gsnbv8[i:i+2], y8_G3_gsnbv8[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x8_G3_gsnbv8[i:i+2], y8_G3_gsnbv8[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x8_G3_gsnbv8)):
    if G3_gsnbv8[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x8_G3_gsnbv8[i], y8_G3_gsnbv8[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv8[i][0]:  # Other True points in grey
        ax.scatter(x8_G3_gsnbv8[i], y8_G3_gsnbv8[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x8_G3_gsnbv8[i], y8_G3_gsnbv8[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv9= [(False, 0.2443), (False, 1.0), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189), (True, 0.0189)]
x9_G3_gsnbv9 = list(range(len(G3_gsnbv9)))
y9_G3_gsnbv9 = [1 - value[1] for value in G3_gsnbv9]
first_true_found = False
for i in range(len(x9_G3_gsnbv9)-1):
    if G3_gsnbv9[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x9_G3_gsnbv9[i:i+2], y9_G3_gsnbv9[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x9_G3_gsnbv9[i:i+2], y9_G3_gsnbv9[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x9_G3_gsnbv9[i:i+2], y9_G3_gsnbv9[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x9_G3_gsnbv9)):
    if G3_gsnbv9[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x9_G3_gsnbv9[i], y9_G3_gsnbv9[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv9[i][0]:  # Other True points in grey
        ax.scatter(x9_G3_gsnbv9[i], y9_G3_gsnbv9[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x9_G3_gsnbv9[i], y9_G3_gsnbv9[i], color=colors_G3_gsnbv[i], marker='x',  s=80)

G3_gsnbv10= [(False, 0.2343), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019), (True, 0.019)]
x10_G3_gsnbv10 = list(range(len(G3_gsnbv10)))
y10_G3_gsnbv10 = [1 - value[1] for value in G3_gsnbv10]

first_true_found = False
for i in range(len(x10_G3_gsnbv10)-1):
    if G3_gsnbv10[i][0] and not first_true_found:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x10_G3_gsnbv10[i:i+2], y10_G3_gsnbv10[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x10_G3_gsnbv10[i:i+2], y10_G3_gsnbv10[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x10_G3_gsnbv10[i:i+2], y10_G3_gsnbv10[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gsnbv[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x10_G3_gsnbv10)):
    if G3_gsnbv10[i][0] and not first_true_found:  # First True point in blue
        ax.scatter(x10_G3_gsnbv10[i], y10_G3_gsnbv10[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gsnbv10[i][0]:  # Other True points in grey
        ax.scatter(x10_G3_gsnbv10[i], y10_G3_gsnbv10[i], color='red', marker='*',  s=Gray_star_size)
    else:  # False points
        ax.scatter(x10_G3_gsnbv10[i], y10_G3_gsnbv10[i], color=colors_G3_gsnbv[i], marker='x',  s=80)







G3_gnbv1 =[(False, 0.3761), (False, 0.1685), (False, 0.3077), (False, 0.3065), (False, 0.1843), 
           (True, 0.1394),(True, 0.1394),(True, 0.1394),(True, 0.1394),(True, 0.1394),(True, 0.1394)]
colors_G3_gnbv1 = ['lime' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_gnbv1]
x1_G3_gnbv1 = list(range(len(G3_gnbv1)))
y1_G3_gnbv1 = [1 - value[1] for value in G3_gnbv1]

first_true_found = False
for i in range(len(x1_G3_gnbv1)-1):
    if G3_gnbv1[i][0] and not first_true_found and  G3_gnbv1[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x1_G3_gnbv1[i:i+2], y1_G3_gnbv1[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x1_G3_gnbv1[i:i+2], y1_G3_gnbv1[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x1_G3_gnbv1[i:i+2], y1_G3_gnbv1[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gnbv1[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x1_G3_gnbv1)):
    if G3_gnbv1[i][0] and not first_true_found and  G3_gnbv1[i][1] <threshold:  # First True point in blue
        ax.scatter(x1_G3_gnbv1[i], y1_G3_gnbv1[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_scnbvp9[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x1_G3_gnbv1[i], y1_G3_gnbv1[i], color='lime', marker='*',  s=Gray_star_size)
    elif G3_scnbvp9[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x1_G3_gnbv1[i], y1_G3_gnbv1[i], color='lime', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x1_G3_gnbv1[i], y1_G3_gnbv1[i], color=colors_G3_gnbv1[i], marker='x',  s=80)        
        


G3_gnbv2 = [(False, 0.3796), (True, 0.2512), (True, 0.2602), (True, 0.2475), (True, min_occlusion),
(True, min_occlusion),(True, min_occlusion),(True, min_occlusion),(True, min_occlusion) ,(True, min_occlusion),(True, min_occlusion)]   
colors_G3_gnbv2 = ['lime' if (not value[0]) or (value[0] > threshold ) else 'grey' for value in G3_gnbv2]
x2_G3_gnbv2 = list(range(len(G3_gnbv2)))
y2_G3_gnbv2 = [1 - value[1] for value in G3_gnbv2]
first_true_found = False
for i in range(len(x2_G3_gnbv2)-1):
    if G3_gnbv2[i][0] and not first_true_found and  G3_gnbv2[i][1] <threshold:  # First True point
        # ax.plot(x1[i:i+2], y1[i:i+2], color='blue')  # Line in blue
        first_true_found = True
        ax.plot(x2_G3_gnbv2[i:i+2], y2_G3_gnbv2[i:i+2], color='gray',linewidth=G3_gsnbv_linewidth,  linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    elif first_true_found:  # Subsequent lines after the first True point
        ax.plot(x2_G3_gnbv2[i:i+2], y2_G3_gnbv2[i:i+2], color='gray', linewidth=G3_gsnbv_linewidth, linestyle='--',alpha=True_line_alpha)  # Dashed line in grey
    else:
        ax.plot(x2_G3_gnbv2[i:i+2], y2_G3_gnbv2[i:i+2], linewidth=G3_gsnbv_linewidth, color=colors_G3_gnbv2[i])  # Normal line
# Adjust scatter points to use '*' for True and 'x' for False, with first True in blue and others in grey
first_true_found = False
for i in range(len(x2_G3_gnbv2)):
    if G3_gnbv2[i][0] and not first_true_found and  G3_gnbv2[i][1] <threshold:  # First True point in blue
        ax.scatter(x2_G3_gnbv2[i], y2_G3_gnbv2[i], color='red', marker='*',  s=terminate_size)
        first_true_found = True
    elif G3_gnbv2[i][0] and  first_true_found:  # Other True points in grey
        ax.scatter(x2_G3_gnbv2[i], y2_G3_gnbv2[i], color='lime', marker='*',  s=Gray_star_size)
    elif G3_gnbv2[i][0] or  first_true_found:  # Other True points in grey
        ax.scatter(x2_G3_gnbv2[i], y2_G3_gnbv2[i], color='lime', marker='*',  s=80)    
    else:  # False points
        ax.scatter(x2_G3_gnbv2[i], y2_G3_gnbv2[i], color=colors_G3_gnbv2[i], marker='x',  s=80)  







# Add horizontal dashed lines at specified y-values
for y_dashed in [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 , 0.7, 0.8, 0.9]:
    ax.axhline(y=y_dashed, color='gray', linestyle='--', linewidth=1)

# Set axis limits and labels
ax.set_xticks(range(11))
# ax.set_yticks(np.arange(-0.01, 1.01, 0.1))
ax.set_yticks(np.arange(-0.1, 1.1, 0.1))
ax.set_xlim([-0.1, 10.1])  # Set limit for x-axis
ax.set_ylim([-0.01, 1.01])  # Set limit for y-axis
ax.set_xlabel('Planning iterations', fontsize=12)
ax.set_ylabel('Unoccluded rate [%]', fontsize=12)
# ax.set_title('Line Chart of Data Points')
# Set larger font size for tick labels
ax.tick_params(axis='both', which='major', labelsize=12)
# Add legend
# Define legend manually to ensure correct styles
legend_elements = [
    mlines.Line2D([], [], color='red', marker='*', linestyle='-', markersize=10, label='GS-NBV Group3'),
    # mlines.Line2D([], [], color='cyan', marker='*', linestyle='-', markersize=10, label='GS-NBV Group2'),
    mlines.Line2D([], [], color='cyan', marker='*', linestyle='-', markersize=10, label='SC-NBVP Group3'),
    mlines.Line2D([], [], color='lime', marker='*', linestyle='-', markersize=10, label='GNBV Group3')
]

# Add the custom legend to the plot
ax.legend(handles=legend_elements, loc='lower right')
plt.savefig('/home/jcd/Pictures/my_plot.png', format='png', dpi=300)
# Show the plot
plt.show()
