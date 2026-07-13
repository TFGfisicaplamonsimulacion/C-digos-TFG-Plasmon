ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/1000) ,['BK7','ag', 'air' ], [ 0., 40. ,0. ] 
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
plt.plot(angulos,R,color="#909090")
maxi=np.argmin(R)
maxii=np.round(R[maxi],3)
plt.plot(angulos[maxi], R[maxi],"kx", label=f"R mín= {maxii[0]}")
plt.title("Reflexión de la placa en función de los ángulos para la plata")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Reflexión ")
plt.show()
plt.close()

ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/1000) ,['BK7','au', 'air' ], [ 0., 40. ,0. ] 
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
plt.plot(angulos,R,color="#a0a000")
maxi=np.argmin(R)
maxii=np.round(R[maxi],3)
plt.plot(angulos[maxi], R[maxi],"kx", label=f"R mín= {maxii[0]}")
plt.title("Reflexión de la placa en función de los ángulos para el oro")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Reflexión ")
plt.show()
plt.close()

ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/1000) ,['BK7','ag', 'air' ], [ 0., 40. ,0. ] 
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
plt.plot(angulos,Abs,color="#909090")
maxi=np.argmin(R)
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"kx", label=f"Abs máx= {maxii[0]}")
plt.title("Absorción de la placa en función de los ángulos para la plata")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("Abs plata aire")
plt.show()
plt.close()

ldo,angulos,materiales,grosores = 633,np.arange(0.,90,90/1000) ,['BK7','au', 'air' ], [ 0., 40. ,0. ] 
r, t, R, T = cr.Multicapa_Matrix(ldo= ldo,tempera=tempera, mats= materiales,d= grosores,theta_i= angulos)
Abs = (1 - R -T.real)
plt.plot(angulos,Abs,color="#a0a000")
maxi=np.argmin(R)
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"kx", label=f"Abs máx= {maxii[0]}")
plt.title("Absorción de la placa en función de los ángulos para el oro")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("Abs oro aire")
plt.show()
plt.close()


plt.figure()
plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],np.array(plasdef),label="Ajuste según la temperatura")
plt.plot(angulos,Abs,label="Valor inicial de la curva")
plt.title("Cambio de la curva de resonancia ")
plt.legend()
plt.xlim(52,56)
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("cambio abs")
plt.show()


vang=round(np.pi*(angula[Nt-1]-angula[0])/Nt/180*1000,4)
plt.figure()
plt.title(f"Evolución de la temperatura a lo largo del tiempo \n Ángulo ={vang} mRad/s ")
plt.xlabel("Ángulo (θ)")
plt.ylabel("${\Delta}T$")
curva=plt.plot(tiempo,ptocentr)
plt.ylim(0,6)

plt.savefig("curva mov")
plt.show()

plt.plot(angulos,Abs,color="#a0a000")
maxi=np.argmin(R)
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"kx", label=f"Abs máx= {maxii[0]}")
maxi=np.argmin(abs(angulos-54))
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"kx", label=f"Abs máx= {maxii[0]}")
maxi=np.argmin(abs(angulos-56))
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"ko", label=f"Abs máx= {maxii[0]}")
plt.title("Absorción de la placa en función de los ángulos para el oro")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("Abs oro aire")
plt.show()
plt.close()

plt.plot(angulos,Abs,color="#909090")
maxi=np.argmin(R)
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"gx", label=f"Abs máx= {maxii[0]}")
maxi=np.argmin(abs(angulos-54))
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"mx", label=f"Abs 54º= {maxii[0]}")
maxi=np.argmin(abs(angulos-56))
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"rx", label=f"Abs 56º= {maxii[0]}")
maxi=np.argmin(abs(angulos-49))
maxii=np.round(Abs[maxi],3)
plt.plot(angulos[maxi], Abs[maxi],"yx", label=f"Abs 49º= {maxii[0]}")
plt.title("Absorbancia de la placa en función de los ángulos para la plata")
plt.legend()
plt.xlabel("Ángulo θ")
plt.ylabel("Absorbancia ")
plt.savefig("Abs oro aire")
plt.show()
plt.close()

plt.figure()
plt.plot(angulos[int(0+len(Abs)*(cl-45/red)/90):int(Nt+len(Abs)*((cl-45/red)/90))],np.array(plasdef),label="Ajuste según la temperatura")
plt.plot(angulos,Abs,label="Valor inicial de la curva")
plt.title("Cambio de la curva de resonancia ")
plt.legend()
plt.xlim(53,56) #Hay que ajustarlo en función de donde esté el pico para poder apreciar bien la diferencia
plt.xlabel("Ángulo θ")
plt.ylabel("Absorción ")
plt.savefig("cambio abs")
plt.show()


        #### CREAMOS LA IMAGEN 2D Y LA GUARDAMOS
        fig, ax = plt.subplots()
        ax = sns.heatmap(placa, linewidth=0,vmin="0",vmax="10" ,cmap="plasma",xticklabels=5,yticklabels=5)
        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        img = Image.open(buf)
        listagif.append(img)

        ### CREAMOS LA IMAGEN 3D Y LA GUARDAMOS 
        fig2 = plt.figure()
        ax2 = plt.axes(projection="3d")
        ax2.plot_surface(X, Y,placa, cmap="plasma",edgecolor="none") 
        ax2.set_zlim(0,10)
        ax2.set_zlabel("Grados")
        plt.gca().invert_yaxis()
        plt.title(f'Ángulo ={ang} θ \n Tiempo = {tiemp} s')
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        plt.show()
        plt.close()
        img2 = Image.open(buf2)
        listagif2.append(img2)

#Creamos los gifs
listagif[0].save(
    "output.gif",
    save_all=True,
    append_images=listagif[1:],
    duration=200,         
    loop=0)             
listagif2[0].save(
    "output2.gif",
    save_all=True,
    append_images=listagif2[1:],
    duration=200,
    loop=0)



maximo=np.max(ptocentr)
for i in range(len(ptocentr)):
    if ptocentr[i]>maximo*(0.3):
        print (i*T_final/len(ptocentr))
        break
    
maximo=np.max(ptocentr)
for i in range(len(ptocentr)):
    if ptocentr[i]>maximo*(0.6):
        print (i*T_final/len(ptocentr))
        break
    
maximo=np.max(ptocentr)
for i in range(len(ptocentr)):
    if ptocentr[i]>maximo*(0.9):
        print (i*T_final/len(ptocentr))
        break
    
maximo=np.max(ptocentr)
for i in range(len(ptocentr)):
    if ptocentr[i]>maximo*(0.95):
        print (i*T_final/len(ptocentr))
        break