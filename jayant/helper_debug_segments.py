
# c = [((a[0]+ b[0])/2, (a[1] + b[1])/2) for a,b in segments]
x = coords[:,0]
y = coords[:,1]
possibilities = []
a = []
b = []
for pt1, pt2 in segments:
	possibilities.append(pt1)
	possibilities.append(pt2)
	a.append(pt1)
	b.append(pt2)

x_p = [px for px,_ in possibilities]
y_p = [py for _,py in possibilities]
plt.clf()
plt.scatter(x, y, color = "blue")
plt.plot(x_p, y_p, color = "orange")
plt.scatter([x for x,y in a], [y for x,y in a], color = 'g')
plt.scatter([x for x,y in b], [y for x,y in b], color = 'r')
plt.scatter([coords[chosen][0]], [coords[chosen][1]], color = 'purple')
# plt.scatter([new_x],[new_y],color = "black")
plt.savefig("junk/temp.png")
# plt.scatter([x for x,y in c], [y for x,y in c], color = 'y')
# plt.scatter([new_x], [new_y] , color = 'y')

# if new_x is not None: